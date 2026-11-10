import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
from .layers import MLP


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
    # 【修复3】去掉 mlp_dim，直接接收 hidden_dim
    def __init__(self, hidden_dim, num_classes, use_position_encoding=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_position_encoding = use_position_encoding

        # MLP 只做维度内的非线性特征聚合，不再做降维，速度极快
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        if self.use_position_encoding:
            self.position_encoding = nn.Parameter(
                torch.randn(1, num_classes, hidden_dim)
            )

    def forward(self, mask_tokens):
        """
        接收来自 Transformer 的 mask_tokens: [B, num_classes, C]
        """
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
    # 获取当前进程的 global rank，如果是非分布式环境则默认为 0
    global_rank = int(os.environ.get("RANK", 0))

    if global_rank == 0:
        print(f"Loading pretrained weights from {pretrained_path}...")

    checkpoint = torch.load(pretrained_path, map_location="cpu")

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

        # 1. 形状完全匹配，直接继承
        if v.shape == model_v.shape:
            updated_dict[k] = v
            loaded_keys.append(k)

        # 2. 完美适配 MaskDownSampler 的首层
        elif k == "memory_encoder.mask_downsampler.encoder.0.weight":
            if v.shape[1] != model_v.shape[1]:
                v_mean = v.mean(dim=1, keepdim=True)
                adapted_v = v_mean.repeat(1, model_v.shape[1], 1, 1)
                updated_dict[k] = adapted_v
                adapted_keys.append(k)
            else:
                updated_dict[k] = v
                loaded_keys.append(k)

        # 3. 完美适配图像特征图尺寸的更改
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

        # 4. Mask Tokens 适配
        elif k == "sam_mask_decoder.mask_tokens.weight":
            min_cls = min(v.shape[0], model_v.shape[0])
            adapted_v = model_v.clone()
            adapted_v[:min_cls, :] = v[:min_cls, :]
            updated_dict[k] = adapted_v
            adapted_keys.append(k)

        # 5. ObjPointer 的位置编码适配
        elif k == "obj_ptr_proj.position_encoding":
            min_cls = min(v.shape[1], model_v.shape[1])
            adapted_v = model_v.clone()
            adapted_v[:, :min_cls, :] = v[:, :min_cls, :]
            updated_dict[k] = adapted_v
            adapted_keys.append(k)

        # 6. 其他不兼容模块
        else:
            skipped_keys.append((k, v.shape, model_v.shape))

    # 7. 为新增类别自动复制超网络 MLP 的权重
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

    # 仅在主进程打印加载结果统计
    if global_rank == 0:
        print("\n" + "=" * 60)
        print(f"✅ Loaded {len(loaded_keys)} exact match weights.")
        print(
            f"✨ Adapted {len(adapted_keys)} weights with intelligent dimension transformations."
        )
        print(
            f"⚠️  Skipped {len(skipped_keys)} weights due to architectural improvements."
        )

        if skipped_keys:
            print("\nSkipped layers:")
            for sk, p_shape, m_shape in skipped_keys:
                print(f"  - {sk}: expected {m_shape}, got {p_shape}")
        print("=" * 60 + "\n")

    return model


class CircularList:
    def __init__(self, max_length):
        """初始化列表类，指定最大长度"""
        self.max_length = max_length
        self.data = []

    def append(self, item):
        self.data.append(item)
        if len(self.data) > self.max_length:
            self.data.pop(0)  # 移除第一个元素（最先加入的数据）

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
