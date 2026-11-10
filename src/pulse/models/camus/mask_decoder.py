# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from ..common.legacy_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,  # 新增：指定类别数量
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        use_high_res_features: bool = True,
        feat_size: Tuple[int, int] = (16, 16),
    ) -> None:
        """
        Predicts labels given an image, using a transformer architecture.
        直接为每个类别输出一个mask。

        Arguments:
          num_classes (int): 类别数量，每个类别对应一个mask
          transformer_dim (int): transformer的通道维度
          transformer (nn.Module): 用于预测mask的transformer
          activation (nn.Module): 上采样mask时使用的激活函数
          iou_head_depth (int): 用于预测mask质量的MLP深度
          iou_head_hidden_dim (int): 用于预测mask质量的MLP隐藏层维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes  # 存储类别数量
        self.feat_size = feat_size
        self.image_pe = nn.Parameter(
            torch.zeros(1, transformer_dim, feat_size[0], feat_size[1])
        )
        # 为每个类别创建一个mask token
        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

        # 输出上采样层
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # 每个类别对应的MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(num_classes)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ):
        """
        Predict labels given image embeddings.

        Arguments:
          image_embeddings (torch.Tensor): 图像编码器输出的特征
          image_pe (torch.Tensor): 位置编码，形状与image_embeddings相同
          repeat_image (bool): 是否在batch维度上重复图像特征

        Returns:
          torch.Tensor: 预测的mask [batch_size, num_classes, H, W]
          torch.Tensor: 包含了高级语义的 mask_tokens_out [batch_size, num_classes, C] (用于指针)
        """
        image_pe = self.image_pe

        # 【修改点1】同时接收 masks 和 mask_tokens_out
        masks, mask_tokens_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # 【修改点2】返回所有类别的mask和特征Token
        return masks, mask_tokens_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ):
        """Predicts labels. See 'forward' for more details."""
        # 获取mask tokens (每个类别一个)
        batch_size = image_embeddings.shape[0]
        output_tokens = self.mask_tokens.weight  # [num_classes, C]
        output_tokens = output_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_classes, C]

        # 在batch维度上扩展
        tokens = output_tokens.expand(batch_size, -1, -1)  # [B, num_classes, c]

        # 在batch维度上重复图像特征（如果需要）
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        # 注意：不再添加dense prompt embeddings

        # 位置编码处理
        assert image_pe.size(0) == 1, "位置编码batch维度应为1"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行transformer
        hs, src = self.transformer(
            image_embedding=src,  # 图像特征 [B, C, H, W]
            image_pe=pos_src,  # 位置编码 [B, C, H, W]
            point_embedding=output_tokens,  # 类别tokens [B, num_classes, C]
        )
        mask_tokens_out = hs  # [B, num_classes, c] （这就是真正的指针来源）

        # 上采样mask embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # 为每个类别生成mask
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_classes, c']
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 【修改点3】同时返回 masks 图像和 mask_tokens_out 指针向量
        return masks, mask_tokens_out
