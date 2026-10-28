"""Vector-form attention utilities used by the ADVEDM-A / ADVEDM-R pipelines."""

import torch
import torch.nn.functional as F


def attention_weights(block, x: torch.Tensor):
    """Softmax attention and values of an OpenAI-CLIP resblock (seq-first x)."""
    x_ln = block.ln_1(x)
    seq_len, batch_size, dim = x_ln.shape
    num_heads = block.attn.num_heads
    head_dim = dim // num_heads
    qkv = F.linear(x_ln, block.attn.in_proj_weight, block.attn.in_proj_bias)
    q, k, v = qkv.reshape(seq_len, batch_size, 3, num_heads, head_dim).permute(2, 1, 3, 0, 4)
    attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * head_dim ** -0.5  # float32 for stability
    attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
    attn_weights = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    return attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8), v


def openai_tokens(vis, images_normalized: torch.Tensor) -> torch.Tensor:
    """OpenAI-CLIP tower input tokens [B, 1 + N, D]: conv1, class token, positional embedding, ln_pre."""
    x = vis.conv1(images_normalized.to(vis.conv1.weight.dtype)).flatten(2).transpose(1, 2)
    x = torch.cat([vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    return vis.ln_pre(x + vis.positional_embedding.to(x.dtype))


def extract_cls_to_patch_attention(vision_encoder, images_normalized: torch.Tensor) -> torch.Tensor:
    """CLS->patch attention per image from an OpenAI-CLIP-style tower, averaged over layers and heads."""
    x = openai_tokens(vision_encoder, images_normalized).permute(1, 0, 2)
    all_cls_attentions = []
    for block in vision_encoder.transformer.resblocks:
        attn_weights, v = attention_weights(block, x)
        seq_len, batch_size, dim = x.shape
        all_cls_attentions.append(attn_weights.mean(dim=1)[:, 0, 1:])

        attn_output = torch.matmul(attn_weights, v.float())
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, dim).transpose(0, 1)
        attn_output = F.linear(
            attn_output.to(dtype=block.attn.out_proj.weight.dtype),
            block.attn.out_proj.weight,
            block.attn.out_proj.bias,
        )

        x = x + attn_output
        x = x + block.mlp(block.ln_2(x))

    return torch.stack(all_cls_attentions, dim=0).mean(dim=0)


def reallocate_attention_vector(
    A_orig: torch.Tensor,
    A_ref: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.4,
) -> torch.Tensor:
    """Reallocate attention per Equation 10 in vector form."""
    if A_orig.shape != A_ref.shape or A_orig.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: A_orig={tuple(A_orig.shape)}, "
            f"A_ref={tuple(A_ref.shape)}, mask={tuple(mask.shape)}"
        )

    return mask * (1 - beta) * A_orig + (1 - mask) * beta * A_ref


def compute_attention_weighted_features_vector(
    A_vec: torch.Tensor,
    patch_embeds: torch.Tensor,
) -> torch.Tensor:
    """Compute attention-weighted patch embeddings A * [patch]."""
    if A_vec.ndim != 2 or patch_embeds.ndim != 3:
        raise ValueError(
            f"Expected A_vec [B,N] and patch_embeds [B,N,D], got "
            f"{tuple(A_vec.shape)} and {tuple(patch_embeds.shape)}"
        )
    if A_vec.shape[0] != patch_embeds.shape[0] or A_vec.shape[1] != patch_embeds.shape[1]:
        raise ValueError(
            f"Shape mismatch: A_vec={tuple(A_vec.shape)}, patch_embeds={tuple(patch_embeds.shape)}"
        )

    return A_vec.unsqueeze(-1) * patch_embeds
