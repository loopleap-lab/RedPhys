"""realm.defend("adpo", attack_dirs, output): trains the victim's vision encoder."""

import os
import time
from pathlib import Path
from typing import ClassVar

from ..base_defense import BaseDefense
from .config import AdPOConfig
from .data import load_pairs, read_jsonl, write_jsonl


def _dtype(c):
    import torch
    return torch.bfloat16 if c.precision == "bf16" and c.device != "cpu" else torch.float32


class AdPODefense(BaseDefense):
    trains: ClassVar[bool] = True
    config: AdPOConfig

    def clean(self, image_path: str, **kwargs):
        raise NotImplementedError("AdPO fine-tunes the victim: serve <output>/checkpoint and evaluate it")

    def fit(self, inputs: list[str], out_dir: Path) -> dict:
        """Answers (rank 0) -> train (all ranks) -> checkpoint; resumable."""
        out, start, main = Path(out_dir), time.time(), int(os.environ.get("RANK", 0)) == 0
        done, failed = out / "responses.done", out / "responses.failed"
        if main:
            failed.unlink(missing_ok=True)
            try:
                self.prepare(inputs, out)
            except BaseException:
                failed.touch()
                raise
            done.touch()
        while not done.exists() or done.stat().st_mtime < start:
            if failed.exists() and failed.stat().st_mtime >= start:
                raise RuntimeError("stage 1 failed on rank 0")
            time.sleep(10)
        visual = out / "visual.safetensors"
        if not visual.exists():
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()  # stage-1 model, else "auto" sees less memory
            self.train(out)
        if main and not (out / "checkpoint" / "config.json").exists():
            from .export import merge
            merge(self.config.model, visual, out / "checkpoint")
        return {"pairs": "pairs.jsonl", "train_rows": len(read_jsonl(out / "responses.jsonl")),
                "visual": visual.name, "checkpoint": "checkpoint"}

    def prepare(self, inputs: list[str], out: Path) -> list[dict]:
        from .models import load
        from .responses import prepare
        c = self.config
        pairs = load_pairs(inputs, c.holdout_attacks, c.test_fraction, c.seed)
        write_jsonl(out / "pairs.jsonl", pairs)
        model, vlm = load(c.model, c.device if c.adv_source == "realm" and not c.server_url else None, _dtype(c))
        return prepare(c, pairs, model and model.eval(), vlm, out / "responses.jsonl")

    def train(self, out: Path) -> Path:
        import torch
        from transformers import TrainingArguments
        from .export import save_visual
        from .models import load
        from .trainer import AdPOTrainer, PairData, freeze
        c = self.config
        rows = read_jsonl(out / "responses.jsonl")
        if not rows:
            raise ValueError(f"{out / 'responses.jsonl'} has no training rows")
        model, vlm = load(c.model, "auto" if c.device == "auto" else "cpu", _dtype(c))
        print(f"AdPO: {len(rows)} rows, {freeze(model, vlm, c.train_projector) / 1e6:.0f}M trainable vision params")
        world = int(os.environ.get("WORLD_SIZE", 1))
        args = TrainingArguments(
            output_dir=str(out / "trainer"), per_device_train_batch_size=c.micro_batch,
            gradient_accumulation_steps=max(1, c.batch_size // (c.micro_batch * world)), learning_rate=c.lr,
            weight_decay=c.weight_decay, num_train_epochs=c.epochs, lr_scheduler_type="constant",
            bf16=_dtype(c) == torch.bfloat16, use_cpu=c.device == "cpu",
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1,
            save_strategy="no", report_to=[], remove_unused_columns=False, seed=c.seed, dataloader_num_workers=2)
        args._n_gpu = 1  # no DataParallel
        trainer = AdPOTrainer(model=model, args=args, train_dataset=PairData(rows, vlm, c), data_collator=list,
                              adpo_cfg=c, vlm=vlm)
        trainer.train()
        if trainer.is_world_process_zero():
            save_visual(trainer.model, vlm, c.train_projector, out / "visual.safetensors")
        return out / "visual.safetensors"
