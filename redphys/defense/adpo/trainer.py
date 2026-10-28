"""AdPO trainer: L = L_PIO + lam * L_AIO (Eq. 8), structured after ADPO's ADPOTrainer."""

import copy
from collections import Counter

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from transformers import Trainer

from .models import answer, collate, encode, swapped, token_logps, trained, unwrap
from .pgd import pgd


class PairData(torch.utils.data.Dataset):
    """chosen (clean, y_w), rejected (adv, y_l), aio (adv, y_w); pgd: built in the step."""

    def __init__(self, rows, vlm, cfg):
        self.rows, self.vlm, self.cfg = rows, vlm, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, vlm = self.rows[i], self.vlm
        clean = vlm.sized(r["clean"], self.cfg.max_pixels)
        if self.cfg.adv_source == "pgd":
            return {"image": clean, "x": to_tensor(clean), "prompt": r["prompt"]}
        adv = vlm.sized(r["adv"], self.cfg.max_pixels)
        return {"chosen": encode(vlm, clean, r["prompt"], r["y_w"], r["w_end"]),
                "rejected": encode(vlm, adv, r["prompt"], r["y_l"], r["l_end"]),
                "aio": encode(vlm, adv, r["prompt"], r["y_w"], r["w_end"])}


def freeze(model, vlm, train_projector: bool) -> int:
    """Encoder (+ projector) trainable in fp32; returns the parameter count."""
    model.requires_grad_(False)
    n = 0
    for m in trained(model, vlm, train_projector).values():
        for name, p in m.float().named_parameters():
            p.requires_grad_(train_projector or not name.startswith(vlm.projector_params))
            n += p.numel() if p.requires_grad else 0
    return n


class AdPOTrainer(Trainer):

    def __init__(self, *args, adpo_cfg, vlm, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg, self.vlm = adpo_cfg, vlm
        self.model_accepts_loss_kwargs = False  # Trainer averages over accumulation
        self.sums, self.count = Counter(), 0
        # pi_ref / phi_org: frozen copies, run on the same batch (margin 0 at step 0)
        self.ref = {n: copy.deepcopy(m).requires_grad_(False).eval()
                    for n, m in trained(self.model, vlm, adpo_cfg.train_projector).items()}

    def _pgd_inputs(self, model, it, device) -> dict:
        """Paper: x_adv (Eq. 4), then y_w / y_l from the current policy."""
        c, vlm, image, prompt, x = self.cfg, self.vlm, it["image"], it["prompt"], it["x"].to(device)
        net = unwrap(model)
        policy = getattr(net.model, vlm.encoder)
        with torch.no_grad():
            target = vlm.features(self.ref[vlm.encoder], vlm.pixel_inputs(x))
        adv = vlm.pixel_inputs(pgd(vlm, policy, target, x, c.pgd_eps / 255, c.pgd_alpha / 255, c.pgd_steps))

        def inputs(pixels=None, response=None, end=True):  # the prompt (+ response) on the clean image, or with the adversarial pixels
            return {k: (pixels or {}).get(k, v).to(device, v.dtype) for k, v in encode(vlm, image, prompt, response, end).items()}

        net.eval()  # KV cache
        y_w, w_end = answer(net, vlm, inputs(), c.max_new_tokens)
        y_l, l_end = answer(net, vlm, inputs(adv), c.max_new_tokens)
        net.train()
        return {"chosen": inputs(response=y_w, end=w_end), "rejected": inputs(adv, y_l, l_end), "aio": inputs(adv, y_w, w_end)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        c, device, n = self.cfg, self.args.device, len(inputs)
        if c.adv_source == "pgd":
            inputs = [self._pgd_inputs(model, it, device) for it in inputs]
        seqs = [it[k] for k in ("chosen", "rejected", "aio") for it in inputs]
        batch = {k: v.to(device) for k, v in collate(seqs, self.vlm.processor.tokenizer.pad_token_id).items()}
        logps, mask = token_logps(model, batch)
        with torch.no_grad(), swapped(model, self.ref):
            ref = token_logps(model, batch)[0].sum(-1)
        seq = logps.sum(-1)
        margin = c.beta * ((seq[:n] - ref[:n]) - (seq[n:2 * n] - ref[n:2 * n]))
        l_pio = -F.logsigmoid(margin).mean()  # Eq. 5
        lp, m = logps[2 * n:], mask[2 * n:]
        l_aio = -(lp.detach().exp() * lp).sum(-1)  # Eq. 7
        if c.aio_reduction == "mean":
            l_aio = l_aio / m.sum(-1).clamp(min=1)
        self.log_adpo = {"l_pio": l_pio.item(), "l_aio": l_aio.mean().item(), "margin": margin.mean().item(),
                         "reward_acc": (margin > 0).float().mean().item()}
        self.sums.update(self.log_adpo)
        self.count += 1
        return l_pio + c.lam * l_aio.mean()  # Eq. 8

    def log(self, logs, *args, **kwargs):
        if self.count:
            logs.update({k: v / self.count for k, v in self.sums.items()})
            self.sums, self.count = Counter(), 0
        super().log(logs, *args, **kwargs)
