"""Memory-augmented SAM2 model for echocardiographic video segmentation."""

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from ..common.backbone.image_encoder import ImageEncoder
from .mask_decoder import MaskDecoder
from ..common.transformer import TwoWayTransformer, RoPEAttention
from .sam2_utils import ObjPointer, get_1d_sine_pe
from .memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from ..common.memory_attention import MemoryAttention, MemoryAttentionLayer
from .position_encoding import PositionEmbeddingSine


class Sam2Vio(nn.Module):
    def __init__(
        self,
        num_classes=3,
        image_size=256,
        num_maskmem=7,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.num_feature_levels = 3
        self.hidden_dim = self.image_encoder.neck.d_model
        self.mem_dim = 64
        self.num_maskmem = num_maskmem
        self.num_classes = num_classes + 1
        self.use_high_res_features_in_sam = True
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )

        self.sam_mask_decoder = MaskDecoder(
            num_classes=self.num_classes,
            transformer_dim=self.hidden_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            feat_size=(self.image_size[0] // 16, self.image_size[1] // 16),
        )

        # Enable semantic object pointers in the memory encoder.
        self.use_obj_ptrs_in_encoder = True
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0

        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
            self.obj_ptr_proj = ObjPointer(
                hidden_dim=self.hidden_dim, num_classes=self.num_classes
            )

        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(self.num_classes, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, kernel_size=3, padding=1),
        )

        if self.num_maskmem > 0:
            self.maskmem_tpos_enc = torch.nn.Parameter(
                torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
            )
            trunc_normal_(self.maskmem_tpos_enc, std=0.02)
            self.memory_attention = MemoryAttention(
                d_model=256,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayer(
                    activation="relu",
                    dim_feedforward=2048,
                    dropout=0.0,
                    pos_enc_at_attn=False,
                    self_attention=RoPEAttention(
                        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.0
                    ),
                    d_model=256,
                    pos_enc_at_cross_attn_keys=True,
                    pos_enc_at_cross_attn_queries=False,
                    cross_attention=RoPEAttention(
                        embedding_dim=256,
                        rope_k_repeat=True,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.0,
                        kv_in_dim=64,
                    ),
                ),
                num_layers=4,
            )
            self.memory_encoder = MemoryEncoder(
                out_dim=64,
                position_encoding=PositionEmbeddingSine(num_pos_feats=64),
                mask_downsampler=MaskDownSampler(
                    kernel_size=3, stride=2, padding=1, num_classes=self.num_classes
                ),
                fuser=Fuser(layer=CXBlock(dim=256), num_layers=2),
            )

    @staticmethod
    def _recent_frame_indices(num_available, max_frames):
        """Return newest-to-oldest frame indices for the bounded memory bank."""
        start = max(0, num_available - max_frames)
        return list(range(start, num_available))[::-1]

    @staticmethod
    def _memory_mask_probabilities(mask_logits):
        """Convert mutually exclusive semantic logits into memory probabilities."""
        return torch.softmax(mask_logits.float(), dim=1)

    def forward(self, x):
        B, nF, C, H, W = x.shape
        x_flat = x.flatten(0, 1)
        output_list = []
        backbone_out = self.forward_image(x_flat)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        imgs_ids = torch.arange(B * nF, device=x.device).reshape(B, nF).T
        for stage_id, img_ids in enumerate(imgs_ids):
            img_ids = img_ids.to(vision_feats[0].device)
            current_vision_feats = [vf[:, img_ids] for vf in vision_feats]
            current_vision_pos_embeds = [pe[:, img_ids] for pe in vision_pos_embeds]
            current_out = self.track_step(
                stage_id,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_list,
                num_frames=nF,
            )
            output_list.append(current_out)

        all_pred_masks_high_res = [
            output_list[key]["pred_masks_high_res"] for key in range(len(output_list))
        ]
        all_pred_masks_high_res = torch.stack(all_pred_masks_high_res, dim=0).transpose(
            0, 1
        )

        return {
            "all_pred_masks_high_res": all_pred_masks_high_res,
        }

    def forward_image(self, img_batch: torch.Tensor):
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def track_step(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_list,
        run_mem_encoder=True,
        num_frames=8,
    ):

        current_out = {"memory_step": int(frame_idx)}

        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]

        pix_feat, fifo_memory_indices = self._prepare_memory_features(
            frame_idx=frame_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_list=output_list,
            num_frames=num_frames,
        )

        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat,
            high_res_features=high_res_features,
        )
        (low_res_masks, high_res_masks, obj_ptr) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["fifo_memory_indices"] = fifo_memory_indices

        if self.num_maskmem > 0:
            self._encode_memory_in_output(
                current_vision_feats,
                feat_sizes,
                run_mem_encoder,
                high_res_masks,
                current_out,
            )
        return current_out

    def _prepare_memory_features(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_list,
        num_frames=8,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        mem_dim = self.mem_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if frame_idx == 0 or self.num_maskmem == 0:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem, None

        to_cat_memory = []
        to_cat_memory_pos_embed = []
        num_obj_ptr_tokens = 0

        # Corrected FIFO: keep recent frames, newest first.  These are true
        # clip indices, which keeps temporal distances valid after eviction.
        recent = self._recent_frame_indices(len(output_list), self.num_maskmem)
        memory_frame_indices = torch.tensor(recent, device=device, dtype=torch.long)
        memory_frame_indices = memory_frame_indices[:, None].expand(-1, B)
        batch_indices = torch.arange(B, device=device)
        candidate_features = torch.stack(
            [item["maskmem_features"] for item in output_list], dim=0
        )
        candidate_pos = torch.stack(
            [item["maskmem_pos_enc"][-1] for item in output_list], dim=0
        )

        for i in range(memory_frame_indices.shape[0]):
            selected_for_batch = memory_frame_indices[i]
            maskmem_features = candidate_features[selected_for_batch, batch_indices].to(
                device, non_blocking=True
            )
            maskmem_pos_enc = candidate_pos[selected_for_batch, batch_indices].to(
                device
            )
            mem_feat_seq = maskmem_features.flatten(2).permute(2, 0, 1)
            spatial_pos = maskmem_pos_enc.flatten(2).permute(2, 0, 1)

            temporal_pos = self.maskmem_tpos_enc[i]
            temporal_pos = temporal_pos.expand(spatial_pos.size(0), B, mem_dim)
            mem_pos_seq = spatial_pos + temporal_pos

            to_cat_memory.append(mem_feat_seq)
            to_cat_memory_pos_embed.append(mem_pos_seq)

        if self.use_obj_ptrs_in_encoder:
            max_ptr_frames = min(
                memory_frame_indices.shape[0], self.max_obj_ptrs_in_encoder
            )
            pointer_frame_indices = memory_frame_indices[:max_ptr_frames]
            if max_ptr_frames > 0:
                # [frames, batch, classes, C] ->
                # [frames * classes * (C / mem_dim), batch, mem_dim].
                # Permuting first is essential; a direct reshape interleaves
                # batch and class tokens when batch size is greater than one.
                candidate_ptrs = torch.stack(
                    [item["obj_ptr"] for item in output_list], dim=0
                )
                obj_ptrs = torch.stack(
                    [
                        candidate_ptrs[pointer_frame_indices[i], batch_indices]
                        for i in range(max_ptr_frames)
                    ],
                    dim=0,
                )
                num_ptr_frames, _, num_ptr_classes, _ = obj_ptrs.shape
                tpos_dim = C
                memory_steps = torch.tensor(
                    [out["memory_step"] for out in output_list],
                    device=device,
                    dtype=torch.float32,
                )
                frame_distances = frame_idx - memory_steps[pointer_frame_indices]
                self.last_pointer_distances = frame_distances[:, 0].long().tolist()
                assert self.last_pointer_distances == list(
                    range(1, min(frame_idx, 7) + 1)
                )
                obj_pos = get_1d_sine_pe(
                    frame_distances / max(1, self.num_maskmem), dim=tpos_dim
                )
                obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                assert C % self.mem_dim == 0
                tokens_per_class = C // self.mem_dim
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).contiguous()
                obj_ptrs = (
                    obj_ptrs.reshape(
                        num_ptr_frames * num_ptr_classes,
                        B,
                        tokens_per_class,
                        self.mem_dim,
                    )
                    .permute(0, 2, 1, 3)
                    .flatten(0, 1)
                )
                obj_pos = (
                    obj_pos[:, None]
                    .expand(-1, num_ptr_classes * tokens_per_class, -1, -1)
                    .reshape(
                        num_ptr_frames * num_ptr_classes * tokens_per_class,
                        B,
                        self.mem_dim,
                    )
                )
                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        fused_feat_seq = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )

        pix_feat_with_mem = fused_feat_seq.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem, memory_frame_indices.detach()

    def _forward_sam_heads(
        self,
        backbone_features,
        high_res_features=None,
    ):
        low_res_masks, mask_tokens_out = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        low_res_masks = low_res_masks.float()
        high_res_masks = self.upsample_block(low_res_masks)

        if self.use_obj_ptrs_in_encoder:
            obj_ptr = self.obj_ptr_proj(mask_tokens_out)
        else:
            obj_ptr = None

        return (
            low_res_masks,
            high_res_masks,
            obj_ptr,
        )

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        run_mem_encoder,
        high_res_masks,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
    ):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        mask_for_mem = self._memory_mask_probabilities(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        return maskmem_features, maskmem_pos_enc

    def predict(self, x):
        nF, C, H, W = x.shape
        # Keep prior outputs so the memory path can select the bounded set of
        # most recent frames in FIFO order for each new frame.
        mem_list = []
        backbone_out = self.forward_image(x)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )
        all_pred_masks_high_res = []

        for img_ids in range(nF):
            img_ids = torch.tensor([img_ids], device=vision_feats[0].device)
            current_vision_feats = [vf[:, img_ids] for vf in vision_feats]
            current_vision_pos_embeds = [pe[:, img_ids] for pe in vision_pos_embeds]
            current_out = self.track_step(
                img_ids,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                mem_list,
            )
            mem_list.append(current_out)
            all_pred_masks_high_res.append(current_out["pred_masks_high_res"])

        all_pred_masks_high_res = torch.cat(all_pred_masks_high_res, dim=0)
        return all_pred_masks_high_res
