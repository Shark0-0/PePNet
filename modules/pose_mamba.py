from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from utils import fps
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .block import Block

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


# @MODELS.register_module()
class PPMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super(PPMamba, self).__init__()
        self.config = config
        self.sort_axes = config.sort_axes
        self.sort_orders = config.sort_orders
        self.scan_mode = config.scan_mode
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.drop_out = config.drop_out

        # # 定义不同的排序策略
        # sorting_strategies = {
        #     # 策略1：身体层次结构（从上到下）
        #     'hierarchical': [
        #         8,    # head
        #         7,    # neck
        #         9, 10, # left_shoulder, right_shoulder
        #         11,12, # left_elbow, right_elbow
        #         13,14, # left_wrist, right_wrist
        #         0,     # pelvis
        #         1, 2,  # left_hip, right_hip
        #         3, 4,  # left_knee, right_knee
        #         5, 6   # left_ankle, right_ankle
        #     ],
        #     # 策略2：对称关节交替
        #     'symmetric_alternating': [
        #         0,     # pelvis
        #         7,     # neck
        #         8,     # head
        #         9, 10, # shoulders
        #         1, 2,  # hips
        #         11,12, # elbows
        #         3, 4,  # knees
        #         13,14, # wrists
        #         5, 6   # ankles
        #     ],
        #     # 策略3：运动优先级（末端->核心）
        #     'motion_priority': [
        #         13,14, # wrists（最常运动）
        #         11,12, # elbows
        #         5, 6,  # ankles
        #         9, 10, # shoulders
        #         3, 4,  # knees
        #         1, 2,  # hips
        #         8,     # head
        #         7,     # neck
        #         0      # pelvis（核心）
        #     ],
        #     # 策略4：躯干到四肢
        #     'torso_to_limbs': [
        #         0,     # pelvis
        #         7,     # neck
        #         8,     # head
        #         9, 10, # shoulders
        #         1, 2,  # hips
        #         11,12, # elbows
        #         3, 4,  # knees
        #         13,14, # wrists
        #         5, 6   # ankles
        #     ]
        # }
        # # 修改后的排序代码
        # self.desired_order = sorting_strategies[config.sorting_strategy]  # 从配置读取策略

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        self.drop_out = nn.Dropout(self.drop_out)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyzs, features, pose):
        B, T, N, _ = xyzs.shape
        _, T_pose, P, _ = pose.shape  # P = 15
        center = xyzs  # [B, T, N, 3]
        group_input_tokens = features  # [B, T, N, 1024]
        pos = self.pos_embed(center)

        # 对pose进行时序下采样，使用与xyzs相同的采样策略
        pose_frames = torch.split(tensor=pose, split_size_or_sections=1, dim=1)
        pose_frames = [torch.squeeze(input=p, dim=1).contiguous() for p in pose_frames]
        pose_frames = [pose_frames[0]] + pose_frames
        new_pose_frames = []
        temporal_kernel_size = 3  # 与P4DConv中相同
        temporal_stride = T_pose // T  # 计算需要的stride以匹配xyzs的帧数

        for t in range(temporal_kernel_size//2, len(pose_frames)-temporal_kernel_size//2, temporal_stride):
            new_pose_frames.append(pose_frames[t])

        pose = torch.stack(tensors=new_pose_frames, dim=1)  # [B, T, P, 3]

        # distances = torch.norm(center.unsqueeze(3) - pose.unsqueeze(2), dim=-1)  # [B, T, N, P]
        # classes = torch.argmin(distances, dim=-1)  # 最小距离的索引即类别
        # min_distances = torch.min(distances, dim=-1).values  # [B, T, N]



        # 创建动态优先级映射（包含所有15个关节）
        # full_priority = []
        # for i in range(15):
        #     if i in self.desired_order:
        #         full_priority.append(self.desired_order.index(i))
        #     else:
        #         full_priority.append(len(self.desired_order))  # 未指定的关节放在最后
        # priority_mapping = torch.tensor(full_priority, device=classes.device)

        # 生成排序键（保持原有逻辑）
        # # sort_key = priority_mapping[classes].float() * 1e5 + min_distances
        # sort_key = priority_mapping[classes].float() * 1e5 + z_coords
        # sorted_keys, sorted_indices = torch.sort(sort_key, dim=-1)  # sorted_indices: [B, T, N]

        z_coords = center[..., 2]  # [B, T, N]
        sort_key_z = z_coords  # 如果只按z坐标排序，可以取消注释这一行
        sorted_keys_z, sorted_indices_z = torch.sort(sort_key_z, dim=-1)

        # 根据排序后的索引重新排列group_input_tokens和pos
        sorted_indices_z = sorted_indices_z.unsqueeze(-1).expand(-1, -1, -1, group_input_tokens.shape[-1])  # [B, T, N, 1024]
        group_input_tokens_z = torch.gather(group_input_tokens, dim=2, index=sorted_indices_z)  # [B, T, N, 1024]

        sorted_indices_pos_z = sorted_indices_z[:, :, :, 0:1].repeat(1, 1, 1, pos.shape[-1])  # [B, T, N, pos_dim]
        pos_z = torch.gather(pos, dim=2, index=sorted_indices_pos_z)  # [B, T, N, pos_dim]

        ##### x
        x_coords = center[..., 0]  # [B, T, N]
        sort_key_x = x_coords  # 如果只按z坐标排序，可以取消注释这一行
        sorted_keys_x, sorted_indices_x = torch.sort(sort_key_x, dim=-1)
        sorted_indices_x = sorted_indices_x.unsqueeze(-1).expand(-1, -1, -1, group_input_tokens.shape[-1])  # [B, T, N, 1024]
        group_input_tokens_x = torch.gather(group_input_tokens, dim=2, index=sorted_indices_x)  # [B, T, N, 1024]
        sorted_indices_pos_x = sorted_indices_x[:, :, :, 0:1].repeat(1, 1, 1, pos.shape[-1])  # [B, T, N, pos_dim]
        pos_x = torch.gather(pos, dim=2, index=sorted_indices_pos_x)  # [B, T, N, pos_dim]

        ##### y
        y_coords = center[..., 1]  # [B, T, N]
        sort_key_y = y_coords  # 如果只按z坐标排序，可以取消注释这一行
        sorted_keys_y, sorted_indices_y = torch.sort(sort_key_y, dim=-1)
        sorted_indices_y = sorted_indices_y.unsqueeze(-1).expand(-1, -1, -1, group_input_tokens.shape[-1])  # [B, T, N, 1024]
        group_input_tokens_y = torch.gather(group_input_tokens, dim=2, index=sorted_indices_y)  # [B, T, N, 1024]
        sorted_indices_pos_y = sorted_indices_y[:, :, :, 0:1].repeat(1, 1, 1, pos.shape[-1])  # [B, T, N, pos_dim]
        pos_y = torch.gather(pos, dim=2, index=sorted_indices_pos_y)  # [B, T, N, pos_dim]

        # group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z],
        #                                dim=1)
        # pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        group_input_tokens = group_input_tokens_z
        pos = pos_z
        # 交叉排列 T 和 N 维度
        # sorted_group_input_tokens = group_input_tokens.permute(0, 2, 1, 3).reshape(B, N * T, -1)  # [B, N*T, 1024]
        # sorted_pos = pos.permute(0, 2, 1, 3).reshape(B, N * T, -1)  # [B, N*T, trans_dim]

        sorted_group_input_tokens = group_input_tokens.reshape(B, N * T, -1)  # [B, N*T, 1024]
        sorted_pos = pos.reshape(B, N * T, -1)  # [B, N*T, trans_dim]

        x = sorted_group_input_tokens
        x = self.drop_out(x)
        x = self.blocks(x, sorted_pos)
        x = self.norm(x)
        ret = x
        # ret_reshaped = ret.view(B, T, N, -1)

        # return ret,ret_reshaped
        return ret
    # ,ret_reshaped
