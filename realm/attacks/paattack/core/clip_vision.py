import torch.nn as nn
import torch.nn.functional as F


class ClipVisionModel(nn.Module):
    """Wraps an open_clip ViT model to return embeddings, tokens, and attention."""

    def __init__(self, model, preprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self._attn_weights = {}
        self._attn_hooks = []

    def _register_attn_hooks(self):
        """Monkey-patch each ResidualAttentionBlock to capture attention weights."""
        self._remove_attn_hooks()
        self._attn_weights = {}
        for i, block in enumerate(self.model.transformer.resblocks):
            def make_hook(blk, idx):
                def hooked_attention(q_x, k_x=None, v_x=None, attn_mask=None):
                    k_x = k_x if k_x is not None else q_x
                    v_x = v_x if v_x is not None else q_x
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(q_x.dtype)
                    attn_output, attn_weight = blk.attn(
                        q_x, k_x, v_x,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                    )
                    self._attn_weights[idx] = attn_weight
                    return attn_output
                return hooked_attention

            self._attn_hooks.append((block, block.attention))
            block.attention = make_hook(block, i)

    def _remove_attn_hooks(self):
        """Restore original attention methods."""
        for block, orig_fn in self._attn_hooks:
            block.attention = orig_fn
        self._attn_hooks = []
        self._attn_weights = {}

    def forward(self, vision, output_normalize, tokens=False, attention=False):
        """Embedding, or (embedding, tokens[, attention per block])."""
        self.model.output_tokens = tokens
        if attention:
            self._register_attn_hooks()
        try:
            out = self.model(self.preprocess(vision))
        finally:
            attentions = self._attn_weights
            self._remove_attn_hooks()  # also on an error
        if output_normalize:
            out = tuple(F.normalize(o, dim=-1) for o in out) if tokens else F.normalize(out, dim=-1)
        return (*out, attentions) if attention and tokens else out
