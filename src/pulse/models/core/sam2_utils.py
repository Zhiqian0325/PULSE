"""Shared neural-network and checkpoint utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
from ..common.layers import MLP


class TemporalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, seq_len, device):
        ids = torch.arange(seq_len, device=device)
        return self.emb(ids)  # [seq_len, dim]


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=128, temperature=10000, normalize=True, scale=2 * 3.1415
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).float()
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)  # C x H x W
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # B x C x H x W
        return pos


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ObjPointer(nn.Module):
    """Project semantic mask tokens into memory object pointers."""

    def __init__(self, hidden_dim, num_classes, use_position_encoding=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_position_encoding = use_position_encoding

        # Aggregate features without reducing the hidden dimension.
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        if self.use_position_encoding:
            self.position_encoding = nn.Parameter(
                torch.randn(1, num_classes, hidden_dim)
            )

    def forward(self, mask_tokens):
        """Project Transformer mask tokens shaped [B, num_classes, C]."""
        obj_ptr = self.mlp(mask_tokens)

        if self.use_position_encoding:
            obj_ptr = obj_ptr + self.position_encoding

        return obj_ptr  # [B, num_classes, C]


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def load_pretrained_weights(model, pretrained_path, strict=False):
    # Default to rank zero outside distributed execution.
    global_rank = int(os.environ.get("RANK", 0))

    if global_rank == 0:
        print(f"Loading pretrained weights from {pretrained_path}...")

    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    updated_dict = {}

    loaded_keys, adapted_keys, skipped_keys = [], [], []

    for k, v in state_dict.items():
        if k not in model_dict:
            continue

        model_v = model_dict[k]

        # Load exact shape matches directly.
        if v.shape == model_v.shape:
            updated_dict[k] = v
            loaded_keys.append(k)

        # Adapt the first MaskDownSampler layer to the semantic channel count.
        elif k == "memory_encoder.mask_downsampler.encoder.0.weight":
            if v.ndim == 4 and model_v.ndim == 4 and v.shape[2:] == model_v.shape[2:]:
                # Official SAM2 consumes a single mask channel. A4C-MemSAM2
                # consumes mutually exclusive semantic probabilities. Reuse
                # the pretrained filters for every input channel and retain
                # the model initialization for any additional output filters.
                source_v = v.mean(dim=1, keepdim=True).repeat(1, model_v.shape[1], 1, 1)
                adapted_v = model_v.clone()
                shared_outputs = min(source_v.shape[0], model_v.shape[0])
                adapted_v[:shared_outputs] = source_v[:shared_outputs]
                updated_dict[k] = adapted_v
                adapted_keys.append(k)
            else:
                skipped_keys.append((k, v.shape, model_v.shape))

        # Resize the learned image positional encoding.
        elif k == "sam_mask_decoder.image_pe":
            if v.shape != model_v.shape:
                adapted_v = F.interpolate(
                    v,
                    size=(model_v.shape[2], model_v.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                updated_dict[k] = adapted_v
                adapted_keys.append(k)

        # Preserve all compatible mask tokens.
        elif k == "sam_mask_decoder.mask_tokens.weight":
            min_cls = min(v.shape[0], model_v.shape[0])
            adapted_v = model_v.clone()
            adapted_v[:min_cls, :] = v[:min_cls, :]
            updated_dict[k] = adapted_v
            adapted_keys.append(k)

        # Preserve compatible object-pointer positional encodings.
        elif k == "obj_ptr_proj.position_encoding":
            min_cls = min(v.shape[1], model_v.shape[1])
            adapted_v = model_v.clone()
            adapted_v[:, :min_cls, :] = v[:, :min_cls, :]
            updated_dict[k] = adapted_v
            adapted_keys.append(k)

        # Record any remaining incompatible tensors.
        else:
            skipped_keys.append((k, v.shape, model_v.shape))

    # Initialize new class hypernetworks from the first class.
    for k in list(model_dict.keys()):
        if "output_hypernetworks_mlps" in k and k not in updated_dict:
            src_k = re.sub(
                r"output_hypernetworks_mlps\.\d+\.", "output_hypernetworks_mlps.0.", k
            )
            if src_k in updated_dict:
                updated_dict[k] = updated_dict[src_k].clone()
                adapted_keys.append(k)

    model_dict.update(updated_dict)
    model.load_state_dict(model_dict, strict=strict)

    # Report checkpoint statistics only from the main process.
    if global_rank == 0:
        print("\n" + "=" * 60)
        print(f"Loaded {len(loaded_keys)} exact-match weights.")
        print(f"Adapted {len(adapted_keys)} weights to the model dimensions.")
        print(f"Skipped {len(skipped_keys)} incompatible weights.")

        if skipped_keys:
            print("\nSkipped layers:")
            for sk, p_shape, m_shape in skipped_keys:
                print(f"  - {sk}: expected {m_shape}, got {p_shape}")
        print("=" * 60 + "\n")

    model.initialization_audit = {
        "exact_loaded_keys": loaded_keys,
        "adapted_keys": adapted_keys,
        "shape_skipped_keys": [
            {"key": k, "source_shape": list(a), "target_shape": list(b)}
            for k, a, b in skipped_keys
        ],
        "not_loaded_keys": [k for k in model_dict if k not in updated_dict],
        "unused_source_keys": [k for k in state_dict if k not in model_dict],
        "module_coverage": {
            prefix: {
                "keys": sum(k.startswith(prefix) for k in model_dict),
                "loaded_keys": sum(k.startswith(prefix) for k in updated_dict),
                "not_loaded_keys": [
                    k
                    for k in model_dict
                    if k.startswith(prefix) and k not in updated_dict
                ],
            }
            for prefix in (
                "image_encoder.",
                "memory_attention.",
                "memory_encoder.",
                "sam_mask_decoder.",
                "obj_ptr_proj.",
                "upsample_block.",
            )
        },
    }
    return model


class CircularList:
    """Bounded FIFO container that evicts the oldest item first."""

    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []

    def append(self, item):
        self.data.append(item)
        if len(self.data) > self.max_length:
            self.data.pop(0)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.data = []
