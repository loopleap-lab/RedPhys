"""Chain of Attack (CoA, CVPR 2025)."""

import itertools
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Dict

import torch
import torchvision.transforms as T
from .core.clipcap import ClipCaptionModel, generate_cap

from ..base_attack import BaseAttack, AttackConfig, AttackResult, LLMClientMixin, pixel_linf, resize_center_crop, to_pil, to_tensor
from ..surrogates import UNIFIED_SURROGATES
from ...data import Sample, file_data_url, image_files
from .data.coa_dataset import CoADataset

CLEAN_PROMPT = "Describe this image in one sentence in English, focusing on the main objects and scene."
TARGET_PROMPT = (
    "Look at this image and write ONE short sentence describing "
    "its main subject as if it were prominently present in a real "
    "scene. Focus on what an observer would notice and how it is "
    "significant — not just its visual appearance.\n\n"
    "Good examples:\n"
    '- Stop sign → "A stop sign is clearly visible ahead."\n'
    '- Cat on a table → "A cat is sitting right in the middle of the table."\n'
    '- Fire extinguisher → "A fire extinguisher is mounted on the wall nearby."\n'
    '- Warning label → "A warning label indicates this area is hazardous."\n\n'
    "Write only the caption, no explanation."
)
GPT_TARGET_PROMPT = (
    "Look at this image and write ONE short sentence describing "
    "its main subject as if it were prominently present in a real scene. "
    "Write only the caption in English, no explanation."
)


@dataclass
class COAAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 100
    alpha: float = 1.0

    clip_model_name: str = "ViT-B/32"
    input_res: int = 224

    clipcap_weights_path: Optional[str] = None
    prefix_length: int = 10

    fusion_type: str = "add_weight"
    a_weight: float = 0.3
    p_neg: float = 0.7

    caption_update_steps: int = 1  # a new ClipCap caption every n steps
    caption_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # auto captions when none are given
    caption_llm_model: str = "gpt-4o-mini"  # fallback captioner
    llm_server_url: Optional[str] = None  # OpenAI-compatible server; None: OpenAI
    llm_api_key_env: str = "OPENAI_API_KEY"

    target_images_dir: Optional[str] = None
    target_captions_path: Optional[str] = None
    clean_captions_path: Optional[str] = None
    clean_images_dir: Optional[str] = None


class COAAttack(LLMClientMixin, BaseAttack):
    config: COAAttackConfig

    def __init__(self, config: COAAttackConfig):
        super().__init__(config)
        config.clipcap_weights_path = config.clipcap_weights_path or str(Path(__file__).parent / "assets" / "conceptual_weights.pt")
        if not Path(config.clipcap_weights_path).exists():
            raise FileNotFoundError(
                f"ClipCap weights not found at {config.clipcap_weights_path}"
            )

        self._loss_models = self._process_vision_info = self._coa_dataset = None  # set last when loaded
        self._to_pil_image = T.ToPILImage()

        self.clean_captions_cache: Dict[str, str] = {}
        self.target_captions_cache: Dict[str, str] = {}
        self._target_caption_warned = False

    def _initialize_models(self):
        """CLIP + ClipCap from the upstream code."""
        if self._loss_models is not None:
            return

        print("Initializing CoA models (CLIP + ClipCap)...")

        import clip
        from transformers import GPT2Tokenizer

        print(f"  Loading CLIP model: {self.config.clip_model_name}")
        self.clip_model, _ = clip.load(
            self.config.clip_model_name, device=self.config.device, jit=False
        )
        self.clip_model.eval().requires_grad_(False)

        # squash, not crop
        res = self.clip_model.visual.input_resolution
        squash = T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_preprocess = T.Compose([squash, T.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0), normalize])
        self.clipcap_preprocess = T.Compose([squash, T.ToTensor(), normalize])

        print(f"  Loading ClipCap from: {self.config.clipcap_weights_path}")
        self.cap_model = ClipCaptionModel(prefix_length=self.config.prefix_length)
        missing = self.cap_model.load_state_dict(
            torch.load(self.config.clipcap_weights_path, map_location="cpu"),
            strict=False,  # GPT-2 attn.bias buffers
        ).missing_keys
        if missing:
            raise RuntimeError(f"ClipCap weights missing keys: {missing[:5]}")
        self.cap_model = self.cap_model.eval().to(self.config.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.clip_module = clip

        # (encode_image, encode_text, tokenize) per surrogate
        clip_tokenize = partial(clip.tokenize, truncate=True)
        loss_models = [(self.clip_model.encode_image, self.clip_model.encode_text, clip_tokenize)]
        if self.config.surrogates == "unified":
            import open_clip
            loss_models = []
            for m in UNIFIED_SURROGATES:
                if m["clip"] == self.config.clip_model_name:
                    mdl, tok = self.clip_model, clip_tokenize
                elif m["clip"]:
                    mdl, _ = clip.load(m["clip"], device=self.config.device, jit=False)
                    tok = clip_tokenize
                else:
                    mdl = open_clip.create_model(m["open_clip"][0], pretrained=m["open_clip"][1], device=self.config.device)
                    tok = open_clip.get_tokenizer(m["open_clip"][0])
                mdl.eval().requires_grad_(False)
                loss_models.append((mdl.encode_image, mdl.encode_text, tok))

        self._loss_models = loss_models  # last: marks the models loaded
        print("✓ CoA models initialized")

    def _initialize_caption_model(self):
        if self._process_vision_info is not None:
            return

        print(f"Initializing {self.config.caption_model} for caption generation...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        self.caption_gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.caption_model,
            dtype=torch.bfloat16,
        ).to(self.config.device)
        self.caption_processor = AutoProcessor.from_pretrained(self.config.caption_model)
        self._process_vision_info = process_vision_info
        print("✓ Caption generation model initialized")

    def _qwen_caption(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        self._initialize_caption_model()
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        text = self.caption_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.caption_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                        return_tensors="pt").to(self.caption_gen_model.device)
        with torch.no_grad():
            ids = self.caption_gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.caption_processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, ids)], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0].strip()

    def _generate_caption(self, image_path: str) -> str:
        if image_path in self.clean_captions_cache:
            return self.clean_captions_cache[image_path]
        caption = self._qwen_caption(image_path, CLEAN_PROMPT, 128)
        if re.search(r'[^\x00-\x7F]', caption):  # non-ASCII (e.g. Qwen's Chinese): OpenAI
            caption = self._gpt_caption(image_path, "clean")
        self.clean_captions_cache[image_path] = caption
        if self.config.clean_captions_path:
            self.save_captions(self.config.clean_images_dir, self.config.clean_captions_path)
        return caption

    def _generate_target_caption(self, image_path: str) -> str:
        key = str(Path(image_path).resolve())  # run.py links one target file once per source
        if key in self.target_captions_cache:
            return self.target_captions_cache[key]
        if not self._target_caption_warned:
            print("\n  WARNING: Auto-generating target captions from target images.\n"
                  "  For stronger attacks, provide manual captions (target_captions_path in the YAML)\n"
                  "  that express adversarial intent (what the VLM should perceive)\n"
                  "  rather than plain visual descriptions.\n")
            self._target_caption_warned = True
        caption = self._qwen_caption(image_path, TARGET_PROMPT, 64).strip('"').strip("'")
        if re.search(r'[^\x00-\x7F]', caption):  # non-ASCII (e.g. Qwen's Chinese): OpenAI
            caption = self._gpt_caption(image_path, "target")
        self.target_captions_cache[key] = caption
        return caption

    def _gpt_caption(self, image_path: str, mode: str = "clean") -> str:
        """Fallback captions from caption_llm_model."""
        client = self.llm()
        text = GPT_TARGET_PROMPT if mode == "target" else CLEAN_PROMPT

        resp = client.chat.completions.create(
            model=self.config.caption_llm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": file_data_url(image_path)}},
                    {"type": "text", "text": text},
                ],
            }],
            max_tokens=128,
        )
        return (resp.choices[0].message.content or "").strip().strip('"').strip("'")

    def save_captions(self, images_dir: str, output_path: str):
        """Clean captions -> output_path, one line per image of images_dir (CoADataset order), up to the first missing one."""
        if not self.clean_captions_cache:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paths = [str(p) for p in image_files(images_dir)]
        old = output_path.read_text().splitlines() if output_path.exists() else []
        known = {**{p: l.strip() for p, l in zip(paths, old) if l.strip()}, **self.clean_captions_cache}
        captions = list(itertools.takewhile(lambda c: c is not None, (known.get(p) for p in paths)))
        output_path.write_text("".join(c + "\n" for c in captions))
        print(f"\n✓ Saved {len(captions)} captions to {output_path} ({len(self.clean_captions_cache)} new)")

    def _clip77(self, text: str) -> str:
        """The longest of the first 200/120/60/30 characters within CLIP's 77 tokens (else text)."""
        for n in (200, 120, 60, 30):
            try:
                self.clip_module.tokenize([text[:n]])
                return text[:n]
            except RuntimeError:
                continue
        return text

    def _run_coa_attack(
        self,
        clean_img_tensor: torch.Tensor,
        clean_caption: str,
        tgt_img_tensor: torch.Tensor,
        tgt_caption: str,
    ) -> torch.Tensor:
        """CoA PGD; adversarial [C, H, W] in [0, 255]."""
        image_org = clean_img_tensor.unsqueeze(0).to(self.config.device)
        image_tgt = tgt_img_tensor.unsqueeze(0).to(self.config.device)

        a = self.config.a_weight
        fusion_type = self.config.fusion_type

        def fuse(img_f, txt_f):
            img_f = img_f / img_f.norm(dim=1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
            if fusion_type == "cat":
                f = torch.cat((img_f, txt_f), dim=1)
            elif fusion_type == "add_weight":
                f = a * img_f + (1 - a) * txt_f
            elif fusion_type == "multiplication":
                f = img_f * txt_f
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")
            return f / f.norm(dim=1, keepdim=True)

        dev = self.config.device
        with torch.no_grad():
            anchors = [
                (fuse(enc_img(self.clip_preprocess(image_org)), enc_txt(tok([clean_caption]).to(dev))),
                 fuse(enc_img(self.clip_preprocess(image_tgt)), enc_txt(tok([tgt_caption]).to(dev))))
                for enc_img, enc_txt, tok in self._loss_models
            ]

        delta = torch.zeros_like(image_org, requires_grad=True)
        epsilon = self.config.epsilon
        alpha = self.config.alpha
        current_caption_list = txt_key = None

        for j in range(self.config.max_iterations):
            adv_image = torch.clamp(image_org + delta, 0, 255)  # as saved

            adv_image_clone = adv_image.clone() / 255.0

            adv_image_clip = self.clip_preprocess(adv_image)

            if j % self.config.caption_update_steps == 0:  # new ClipCap caption
                image_pil = self._to_pil_image((adv_image_clone[0].clamp(0, 1) * 255).byte().cpu())  # ToPILImage's float->uint8, on GPU
                processed_image = self.clipcap_preprocess(image_pil).unsqueeze(0).to(self.config.device)

                with torch.no_grad():
                    prefix = self.clip_model.encode_image(processed_image).to(
                        self.config.device, dtype=torch.float32
                    )
                    prefix_embed = self.cap_model.clip_project(prefix).reshape(
                        1, self.config.prefix_length, -1
                    )

                current_caption_list = [generate_cap(self.cap_model, self.tokenizer, embed=prefix_embed)]

            p_neg = self.config.p_neg
            margin = 1 - p_neg
            if current_caption_list != txt_key:  # text features change only with the caption
                with torch.no_grad():
                    txt_feats = [enc_txt(tok(current_caption_list).to(dev)) for _, enc_txt, tok in self._loss_models]
                txt_key = current_caption_list
            losses = []
            for (enc_img, _, _), txt, (cle_fused_embedding, tgt_fused_embedding) in zip(self._loss_models, txt_feats, anchors):
                cur_fused_embedding = fuse(enc_img(adv_image_clip), txt)
                embedding_sim1 = torch.mean(torch.sum(cur_fused_embedding * cle_fused_embedding, dim=1))
                embedding_sim2 = torch.mean(torch.sum(cur_fused_embedding * tgt_fused_embedding, dim=1))
                losses.append(torch.mean(torch.relu(embedding_sim2 - p_neg * embedding_sim1 + margin)))
            loss = sum(losses) / len(losses)
            loss.backward()

            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()

            if j % 20 == 0:
                print(f"  PGD step {j}/{self.config.max_iterations}: loss={loss.item():.4f}, caption={current_caption_list[0][:50]}")

        adv_image_final = image_org + delta
        adv_image_final = torch.clamp(adv_image_final, 0, 255)
        return adv_image_final.squeeze(0).cpu()

    def generate(self, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        if self._coa_dataset is None:
            self._coa_dataset = CoADataset(
                clean_images_dir=self.config.clean_images_dir,
                clean_captions_path=self.config.clean_captions_path,
                target_images_dir=self.config.target_images_dir,
                target_captions_path=self.config.target_captions_path,
            )
        coa_sample = self._coa_dataset.get(str(sample.id))  # by stem, not sample_idx

        clean_tensor = to_tensor(sample.images[0], 255)[0]  # as loaded (EXIF-upright), the image the VLM sees
        target_tensor = to_tensor(resize_center_crop(coa_sample["target_image"], self.config.input_res), 255)[0]

        meta = sample.metadata  # captions file, else labels (short, within CLIP's 77 tokens), else generated
        clean_caption = coa_sample.get("clean_caption") or meta.get("attack_source_text")
        target_caption = coa_sample.get("target_caption") or meta.get("attack_target_text")
        clean_auto, target_auto = not clean_caption, not target_caption
        if clean_auto:
            clean_caption = self._generate_caption(coa_sample["image_path"])
        if target_auto:
            target_caption = self._generate_target_caption(coa_sample["target_image_path"])
        clean_caption, target_caption = self._clip77(clean_caption), self._clip77(target_caption)

        adv_tensor = self._run_coa_attack(
            clean_tensor,
            clean_caption,
            target_tensor,
            target_caption,
        )

        adv_image = to_pil(adv_tensor, 255)
        return AttackResult(
            adversarial_sample=adv_image,
            perturbation_norm=pixel_linf(adv_image, sample.images[0]),
            metadata={
                "clean_caption": clean_caption,
                "target_caption": target_caption,
                "clean_caption_auto": clean_auto,
                "target_caption_auto": target_auto,
            },
        )
