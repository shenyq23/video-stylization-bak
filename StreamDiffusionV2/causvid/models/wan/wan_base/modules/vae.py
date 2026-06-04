# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import numpy as np
import sys
import json
import cv2

import torch
import torch.cuda as amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from einops import rearrange
from abc import abstractmethod, ABC
import re
from collections import OrderedDict
from PIL import Image
import time


# --- Add necessary paths to find project-specific modules ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../deps/gmflow"))

from deps.sige3d import SIGECausalConv3d, Gather3d, Scatter3d, \
    ScatterWithBlockResidual3d, ScatterGather3d, SIGEModel3d, SIGEModule3d, \
    SIGEConv2d, Gather2d, Scatter2d

from utils.vae_utils.mem_stats import collect_scatter_cache_modules, feat_map_nbytes, format_bytes, scatter_cache_nbytes
from utils.vae_utils.op_time_stats import install_sige_op_time_stats, print_sige_op_time_stats
from utils.vae_utils.mask_utils import build_gather_block_masks, dilate_mask, downsample_mask
from utils.optical_wrapper import GMFlowWrapper, X265MVWrapper, OcclusionComputation  # noqa: E402

from causvid.profiling import PROFILER  # opt-in per-submodule timing (no-op when disabled)


from debugUtil import enable_custom_repr
enable_custom_repr()

# install_sige_op_time_stats()

repo_root = "/media/cephfs/video/VideoUsers/thu2025/zhurui11/StreamDiffusionV2"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(THIS_DIR, "assets")
REPO_DIR = os.path.dirname(THIS_DIR)


def cuda_time_s(fn, *args, **kwargs):
    """
    在 GPU 上精确计时一个函数的执行时间
    返回: seconds (float)
    """
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    fn(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()
    # return start.elapsed_time(end) * 1e-3   # ms -> s
    return start.elapsed_time(end)   # ms


class VAEInterface(ABC, torch.nn.Module):
    @abstractmethod
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        A method to decode a latent representation to an image or video.
        Input: a tensor with shape [B, F // T, C, H // S, W // S] where T and S are temporal and spatial compression factors.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        """
        pass


__all__ = [
    'WanVAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 因为 nn.Conv3d.padding 的顺序是 (T, H, W)，
        # 而 F.pad 对 5D Tensor 的 padding 顺序是 (W, H, T)（而且是成对的）
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1],
                         2 * self.padding[0], 0)
        # nn.Conv3d内部会自动pad, 要关掉
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)   # 默认就是零填充

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False, eps=1e-6):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.eps = eps

        # 这些 nn.Parameter 在推理（inference）时，正常情况下就是从训练好的权重里加载进来的
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim=1, eps=self.eps) * self.scale \
            * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


time_list = []
class Resample(SIGEModule3d):
    def __init__(self, dim, resample_mode, block_size=6):
        assert resample_mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                                 'downsample3d')
        super().__init__()
        self.dim = dim
        self.resample_mode = resample_mode

        # layers
        if resample_mode == 'upsample2d':
            # self.resample = nn.Sequential(
            #     Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
            #     nn.Conv2d(dim, dim // 2, 3, padding=1))

            self.spatupsample = Upsample(
                scale_factor=(2., 2.), mode='nearest-exact')
            self.conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            # self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

            self.gather2d = Gather2d(self.conv, block_size=block_size)
            self.scatter2d = Scatter2d(self.gather2d)

        elif resample_mode == 'upsample3d':
            # self.resample = nn.Sequential(
            #     Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
            #     nn.Conv2d(dim, dim // 2, 3, padding=1))

            self.spatupsample = Upsample(
                scale_factor=(2., 2.), mode='nearest-exact')
            self.conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            # self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

            self.gather2d = Gather2d(self.conv, block_size=block_size)
            self.scatter2d = Scatter2d(self.gather2d)

            self.time_conv = SIGECausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
            self.gather3d = Gather3d(self.time_conv, block_size=block_size)
            self.scatter3d = Scatter3d(self.gather3d)

        elif resample_mode == 'downsample2d':
            # self.resample = nn.Sequential(
            #     nn.ZeroPad2d((0, 1, 0, 1)), # 右边、下边补 1，保证 stride=2 时尺寸整齐（常见 trick
            #     nn.Conv2d(dim, dim, 3, stride=(2, 2)))

            self.downpad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv = SIGEConv2d(dim, dim, 3, stride=(2, 2))
            # self.conv = nn.Conv2d(dim, dim, 3, stride=(2, 2))
            self.gather2d = Gather2d(self.conv, block_size=block_size)
            self.scatter2d = Scatter2d(self.gather2d)

        elif resample_mode == 'downsample3d':
            # self.resample = nn.Sequential(
            #     nn.ZeroPad2d((0, 1, 0, 1)),
            #     nn.Conv2d(dim, dim, 3, stride=(2, 2)))

            self.downpad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv = SIGEConv2d(dim, dim, 3, stride=(2, 2))
            self.gather2d = Gather2d(self.conv, block_size=block_size)
            self.scatter2d = Scatter2d(self.gather2d)

            self.time_conv = SIGECausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            self.gather3d = Gather3d(self.time_conv, block_size=block_size)
            self.scatter3d = Scatter3d(self.gather3d)

    def first_forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        self.spatial_res = (int(h), int(w))
        if self.resample_mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)

                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    # 把时间维度扩展为2倍, 用通道维翻倍（2C）换来时间维翻倍（2T）
                    # 和空间上采样直接用Upsample(scale_factor=(2., 2.)不一样
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack(
                        (x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), dim=3)
                    x = x.reshape(b, c, t * 2, h, w)

        # 统一做空间 2D resample
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        if self.resample_mode == 'upsample3d' or self.resample_mode == 'upsample2d':
            x = self.spatupsample(x)
            x = self.conv(x)    # 2D卷积

        if self.resample_mode == 'downsample2d' or self.resample_mode == 'downsample3d':
            x = self.downpad(x)
            x = self.conv(x)    # 2D卷积，通过stride=2下采样

        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.resample_mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def sparse_forward(self, x, feat_cache=None, feat_idx=[0]):
        if self.resample_mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()

                    x = self.gather3d(x)

                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)

                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        cache = feat_cache[idx]
                        cache = self.gather3d(cache, is_cache_gather=True)
                        x = self.time_conv(x, cache)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    # 把时间维度扩展为2倍, 用通道维翻倍（2C）换来时间维翻倍（2T）
                    N, C2, T, bh, bw = x.shape
                    assert C2 % 2 == 0
                    C = C2 // 2

                    # 拆成 2 份通道
                    # [N, 2, C, T, bh, bw]
                    x = x.view(N, 2, C, T, bh, bw)

                    # 把“2”插到时间维，变成 2T
                    # [N, C, T, 2, bh, bw]
                    x = x.permute(0, 2, 3, 1, 4, 5)
                    # [N, C, 2T, bh, bw]
                    x = x.reshape(N, C, T * 2, bh, bw)

                    x = self.scatter3d(x)

        # 统一做空间 2D resample
        t = x.shape[2]
        # x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()

        if self.resample_mode == 'upsample3d' or self.resample_mode == 'upsample2d':
            x = self.spatupsample(x)
            self.spatial_res = (int(x.size(2)), int(x.size(3)))


            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()
            x = self.gather2d(x, time_list)
            # end.record()
            # torch.cuda.synchronize()
            # time = start.elapsed_time(end)   # ms
            # if self.mode == "sparse":
            # gather2d_time_list.append(time)

            # x = self.gather2d(x.float()).to(torch.bfloat16)
            x = self.conv(x)    # 2D卷积
            x = self.scatter2d(x)

        if self.resample_mode == 'downsample2d' or self.resample_mode == 'downsample3d':
            # TODO: 稀疏计算的时候是不是不需要padding
            # DONE: 不需要
            
            # print(f"Before gather2d: x.mean(): {x.mean():.4f}")
            x = self.gather2d(x, time_list)
            # print(f"After gather2d: x.mean(): {x.mean():.4f}")

            if self.mode == "full" or self.gather2d.mode == "nocache":
                x = self.downpad(x)
            x = self.conv(x)    # 2D卷积，通过stride=2下采样
            # print(f"After conv: x.mean(): {x.mean():.4f}")
            self.spatial_res = (int(x.size(2)), int(x.size(3)))

            x = self.scatter2d(x)
            # print(f"After scatter2d: x.mean(): {x.mean():.4f}")


        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.resample_mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    # 先因果缓存，再gather，因为这个因果缓存要缓存整张图，而不是稀疏的部分
                    cache_x = x[:, :, -1:, :, :].clone()

                    # print(f"Before gather3d: x.mean(): {x.mean():.4f}")
                    x = self.gather3d(x)
                    # print(f"After gather3d: x.mean(): {x.mean():.4f}")

                    cache = feat_cache[idx][:, :, -1:, :, :]
                    # print(f"Before gather3d: cache.mean(): {cache.mean():.4f}")
                    cache = self.gather3d(cache, is_cache_gather=True)
                    # print(f"After gather3d: cache.mean(): {cache.mean():.4f}")
                    # x = self.time_conv(
                    #     torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    x = self.time_conv(torch.cat([cache, x], 2))
                    # print(f"After time_conv: x.mean(): {x.mean():.4f}")

                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = self.scatter3d(x)
                    # print(f"After scatter3d: x.mean(): {x.mean():.4f}")
        return x

    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        if is_first_frame:
            return self.first_forward(x, feat_cache, feat_idx)
        else:
            return self.sparse_forward(x, feat_cache, feat_idx)



cnt = 0
class ResidualBlock(SIGEModule3d):
    def __init__(self, in_dim, out_dim, dropout=0.0, main_block_size=6, shortcut_block_size=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 各个模块得分开写
        self.norm1 = RMS_norm(in_dim, images=False)
        self.nonlinearity = nn.SiLU()
        self.conv1 = SIGECausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.conv2 = SIGECausalConv3d(out_dim, out_dim, 3, padding=1)

        self.main_gather = Gather3d(
            self.conv1, main_block_size, activation_name="silu", rms_norm=self.norm1)
        self.scatter_gather = ScatterGather3d(
            self.main_gather, activation_name="silu", rms_norm=self.norm2)
        # self.scatter1 = Scatter3d(self.main_gather)
        # self.gather2 = Gather3d(self.conv2, main_block_size, activation_name="silu", rms_norm=self.norm2)

        if self.in_dim != self.out_dim:
            # shortcut是1×1×1的卷积，不需要看过去帧，也就不需要类似于WAN的cache_pad
            self.shortcut = SIGECausalConv3d(
                in_dim, out_dim, kernel_size=1, stride=1, padding=0)
            self.shortcut_gather = Gather3d(self.shortcut, shortcut_block_size)
            self.scatter = ScatterWithBlockResidual3d(
                self.main_gather, self.shortcut_gather)
        else:
            self.scatter = Scatter3d(self.main_gather)

        # 兼容wan_vae在conv之前的cache
        self.scatter_for_time1 = Scatter3d(self.main_gather)
        self.scatter_for_time2 = Scatter3d(self.main_gather)

    def first_forward(self, x, feat_cache=None, feat_idx=[0]):
        self.spatial_res = (int(x.size(3)), int(x.size(4)))
        h = x
        if self.in_dim != self.out_dim:
            h = self.shortcut(h)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        cache_x = x[:, :, -CACHE_T:, :, :].clone()

        # 对conv特殊处理
        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device), cache_x
            ],
                dim=2)

        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        x = self.norm2(x)
        x = self.nonlinearity(x)

        cache_x = x[:, :, -CACHE_T:, :, :].clone()

        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device), cache_x
            ],
                dim=2)

        x = self.conv2(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        return x + h

    def sparse_forward(self, x, feat_cache=None, feat_idx=[0]):
        self.spatial_res = (int(x.size(3)), int(x.size(4)))
        # 注意：Gather3d / ScatterGather3d 在 mode=="full" 时是 identity（只记录分辨率/缓存），
        # 不会执行 RMSNorm/激活融合逻辑；因此 full 模式下需要走“dense 计算 + 写 baseline cache”的路径，
        # 否则输出和 cache（feat_cache / scatter original_outputs）都会错，后续 sparse 也会基于错误 baseline。
        if self.mode in ["full", "nocache"]:
            h = x
            if self.in_dim != self.out_dim:
                h = self.shortcut_gather(h)
                h = self.shortcut(h)

            x = self.norm1(x)
            x = self.nonlinearity(x)

            # 记录 input_res，供后续 set_masks 使用
            x = self.main_gather(x)
            cache_x = self.scatter_for_time1(x)[:, :, -CACHE_T:, :, :].clone()

            idx = feat_idx[0]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            x = self.scatter_gather(x)
            # x = self.scatter1(x)
            # x = self.gather2(x)

            x = self.norm2(x)
            x = self.nonlinearity(x)

            cache_x = self.scatter_for_time2(x)[:, :, -CACHE_T:, :, :].clone()

            idx = feat_idx[0]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                )
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            x = self.scatter(x, h)

            return x

        global cnt
        h = x
        if self.in_dim != self.out_dim:
            h = self.shortcut_gather(x)
            h = self.shortcut(h)

        # print(f"{cnt}: Enter: {x.mean():.4f}")
        cnt += 1
        # gather自带rms_norm, 激活函数silu
        x = self.main_gather(x)
        # print(f"{cnt}: After Gather: {x.mean():.4f}")
        cnt += 1

        cache_x = self.scatter_for_time1(x)[:, :, -CACHE_T:, :, :].clone()

        # 对conv特殊处理
        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
            )

        # print(f"{cnt}: Before Gather Cache: {feat_cache[idx].mean():.4f}")
        # 这个gather不需要norm和激活，只是简单的gather即可
        cache = self.main_gather(feat_cache[idx], is_cache_gather=True)
        # print(f"{cnt}: After Gather Cache: {cache.mean():.4f}")
        x = self.conv1(x, cache)
        # print(f"{cnt}: After Conv1: {x.mean():.4f}")
        cnt += 1

        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        # gather自带激活函数silu
        x = self.scatter_gather(x)
        # print(f"{cnt}: After Scatter Gather: {x.mean():.4f}")
        cnt += 1   
        # x = self.scatter1(x)
        # if not x.is_contiguous():
        # pass
        # x = self.gather2(x, test_gather3d=True)

        cache_x = self.scatter_for_time2(x)[:, :, -CACHE_T:, :, :].clone()

        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
            )

        # print(f"{cnt}: Before Gather Cache: {feat_cache[idx].mean():.4f}")
        # 这个gather不需要norm和激活，只是简单的gather即可
        cache = self.main_gather(feat_cache[idx], is_cache_gather=True)
        # print(f"{cnt}: After Gather Cache: {cache.mean():.4f}")

        x = self.conv2(x, cache)
        # print(f"{cnt}: After Conv2: {x.mean():.4f}")

        cnt += 1
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        # no sparse: x:[1, 128, 512, 1024], self.scatter = Scatter3d
        # sparse: x:[405, 256, 4, 4]), self.scatter = ScatterWithBlockResidual3d
        x = self.scatter(x, h)
        # print(f"{cnt}: After Scatter: {x.mean():.4f}")
        cnt += 1
        return x

    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        if is_first_frame:
            return self.first_forward(x, feat_cache, feat_idx)
        else:
            return self.sparse_forward(x, feat_cache, feat_idx)


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(SIGEModel3d):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0,
                 resample_block_size=4,
                 residual_main_block_size=4,
                 residual_shortcut_block_size=2):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # 不进行稀疏计算
        # CausalConv3d(3, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(
                    ResidualBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        main_block_size=residual_main_block_size,
                        shortcut_block_size=residual_shortcut_block_size,
                    )
                )
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(
                    Resample(out_dim, resample_mode=mode, block_size=resample_block_size)
                )
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(
                out_dim,
                out_dim,
                dropout,
                main_block_size=residual_main_block_size,
                shortcut_block_size=residual_shortcut_block_size,
            ),
            AttentionBlock(out_dim),
            ResidualBlock(
                out_dim,
                out_dim,
                dropout,
                main_block_size=residual_main_block_size,
                shortcut_block_size=residual_shortcut_block_size,
            ),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            # 不进行稀疏计算
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        # Profiler buckets are named by the *actual* feature-map height at each stage,
        # so labels reflect the real input resolution: e.g. 480/240/120/60 for a
        # 480x832 run, or 512/256/128/64 for a square 512 run (the paper's setting).
        # The record name is evaluated on entry, so a Resample layer is attributed to
        # the (higher) resolution it consumes before it downsamples.
        # 针对conv1的, 不进行稀疏计算
        with PROFILER.record(f"VAE {x.shape[-2]}"):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            with PROFILER.record(f"VAE {x.shape[-2]}"):
                if feat_cache is not None:
                    # if isinstance(layer, Resample):
                        # print("Resample forward")
                    x = layer(x, feat_cache, feat_idx, is_first_frame)
                else:
                    x = layer(x)

        # print("*" * 40)
        # print("Enter middle blocks")
        # middle + head run at the deepest (lowest) resolution
        _mid_head = PROFILER.record(f"VAE {x.shape[-2]}")
        _mid_head.__enter__()
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                # transformers
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end   = torch.cuda.Event(enable_timing=True)
                # start.record()

                x = layer(x)

                # end.record()
                # torch.cuda.synchronize()
                # print(f"encoder transformers: {start.elapsed_time(end):.2f} ms")   # ms

        # head
        # 针对head中的conv, 不进行稀疏计算
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        _mid_head.__exit__(None, None, None)
        return x


class Decoder3d(SIGEModel3d):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0,
                 resample_block_size=4,
                 residual_main_block_size=4,
                 residual_shortcut_block_size=2):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        # 不进行稀疏计算
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(
                dims[0],
                dims[0],
                dropout,
                main_block_size=residual_main_block_size,
                shortcut_block_size=residual_shortcut_block_size,
            ),
            AttentionBlock(dims[0]),
            ResidualBlock(
                dims[0],
                dims[0],
                dropout,
                main_block_size=residual_main_block_size,
                shortcut_block_size=residual_shortcut_block_size,
            ),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(
                    ResidualBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        main_block_size=residual_main_block_size,
                        shortcut_block_size=residual_shortcut_block_size,
                    )
                )
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(
                    Resample(out_dim, resample_mode=mode, block_size=resample_block_size)
                )
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            # 不进行稀疏计算
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                # transformers
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end   = torch.cuda.Event(enable_timing=True)
                # start.record()

                x = layer(x)

                # end.record()
                # torch.cuda.synchronize()
                # print(f"decoder transformers: {start.elapsed_time(end):.2f} ms")   # ms

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, (CausalConv3d, SIGECausalConv3d)):
            count += 1
    return count


flow_time_enc = []
flow_time_dec = []

class WanVAE_(nn.Module):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0,
                 resample_block_size=4,
                 residual_main_block_size=4,
                 residual_shortcut_block_size=2,
                 full_refresh_every=0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]  # 反转序列
        self.full_refresh_every = int(full_refresh_every)
        if self.full_refresh_every < 0:
            raise ValueError("full_refresh_every must be >= 0.")
        self._enc_sparse_call_count = 0
        self._dec_sparse_call_count = 0

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout,
                                 resample_block_size=resample_block_size,
                                 residual_main_block_size=residual_main_block_size,
                                 residual_shortcut_block_size=residual_shortcut_block_size)
        # 这两个也不稀疏计算
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout,
                                 resample_block_size=resample_block_size,
                                 residual_main_block_size=residual_main_block_size,
                                 residual_shortcut_block_size=residual_shortcut_block_size)
        self.first_encode = True
        self.first_decode = True
        self.decoder_sparse_max_res = None
    
    def stream_encode(self, x, scale, mask=None, flow=None, is_nocache=False):
        # self.clear_cache()
        # cache
        t = x.shape[2]
        if self.first_encode:
            self.first_encode = False
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                is_first_frame=True,
            )
            self._enc_conv_idx = [0]

            if is_nocache:
                self.encoder.set_mode("nocache")
            else:
                # 第一次只cache，没有flow的映射和mask
                self.encoder.set_mode("full")

            out_ = self.encoder(
                x[:, :, 1:, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

        else:
            self._enc_sparse_call_count += 1
            refresh_now = (
                self.full_refresh_every > 0 and
                self._enc_sparse_call_count % self.full_refresh_every == 0
            )

            if refresh_now:
                self.encoder.set_mode("full")
                print("*" * 40)
                print(
                    f"[Encoder] periodic full refresh: every {self.full_refresh_every} calls "
                    f"(current={self._enc_sparse_call_count})"
                )
                print("*" * 40)
                self._enc_conv_idx = [0]
                out = self.encoder(
                    x,
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                # set_maks
                # 通过光流/运动向量映射来改变scatter's的original_outputs
                if is_nocache:
                    self.encoder.set_mode("nocache")
                else:
                    self.encoder.set_mode("sparse")
                    print("*" * 40)
                    print("Run in sparse mode!!!")
                    print("*" * 40)
                    # Split into two buckets: "Set Masks" builds the per-layer
                    # gather/scatter indices (really part of each conv stage's sparse
                    # setup), "Cache Warp" is the actual optical-flow warp of the caches.
                    with PROFILER.record("Set Masks"):
                        self.encoder.set_masks(mask)
                    with PROFILER.record("Cache Warp"):
                        self.encoder.flow_cache(flow)

                out = []
                for i in range(t//4):   # 实际就运行一次，因为t=4
                    self._enc_conv_idx = [0]
                    out.append(self.encoder(
                        x[:, :, i*4:(i+1)*4, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    ))
                out = torch.cat(out, 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        # self.clear_cache()
        return mu

    def stream_decode(self, z, scale, mask=None, flow=None):
        assert mask is None and flow is None, "Currently mask and flow are not supported in decoder."
        
        # z: [b,c,t,h,w]
        t = z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        if self.first_decode:
            self.first_decode = False
            self.clear_cache_decode()
            self.first_batch = False
            self._dec_conv_idx = [0]
            out = self.decoder(
                x[:, :, :1, :, :],
                feat_cache=self._dec_feat_map,
                feat_idx=self._dec_conv_idx,
                is_first_frame=True,
            )

            self._dec_conv_idx = [0]
            # Build decoder caches only for sparse-enabled resolutions.
            # self.decoder.set_mode("full")
            self.decoder.set_mode("nocache")

            out_ = self.decoder(
                x[:, :, 1:, :, :],
                feat_cache=self._dec_feat_map,
                feat_idx=self._dec_conv_idx,
            )
            out = torch.cat([out, out_], 2)
        else:
            self._dec_sparse_call_count += 1
            refresh_now = (
                self.full_refresh_every > 0 and
                self._dec_sparse_call_count % self.full_refresh_every == 0
            )

            if refresh_now:
                self.decoder.set_mode("full")
                print("*" * 40)
                print(
                    f"[Decoder] periodic full refresh: every {self.full_refresh_every} calls "
                    f"(current={self._dec_sparse_call_count})"
                )
                print("*" * 40)
                self._dec_conv_idx = [0]
                out = self.decoder(
                    x,
                    feat_cache=self._dec_feat_map,
                    feat_idx=self._dec_conv_idx,
                )
            else:
                # set_masks
                # change scatter's original_outputs through flow
                # self.decoder.set_mode("sparse")
                self.decoder.set_mode("nocache")

                # self.decoder.set_masks(mask)
                # self.decoder.flow_cache(flow)
                # t = cuda_time_s(self.decoder.set_masks, mask)
                # print(f"[TIMING] decoder.set_masks = {t:.2f} ms")

                # t = cuda_time_s(self.decoder.flow_cache, flow)
                # print(f"[TIMING] encoder.flow_cache = {t:.2f} ms")

                out = []
                for i in range(t):  # 实际就运行一次，因为t=1
                    self._dec_conv_idx = [0]
                    out.append(self.decoder(
                        x[:, :, i:(i+1), :, :],
                        feat_cache=self._dec_feat_map,
                        feat_idx=self._dec_conv_idx,
                    ))
                out = torch.cat(out, 2)
        # self.clear_cache()
        return out
    def clear_cache_decode(self):
        self._dec_conv_num = count_conv3d(self.decoder)
        self._dec_conv_idx = [0]
        self._dec_feat_map = [None] * self._dec_conv_num
        self._dec_sparse_call_count = 0

    def clear_cache_encode(self):
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        self._enc_sparse_call_count = 0


def map_wanvae_ckpt_keys(sd: dict) -> dict:
    """
    1) ResidualBlock:  ...residual.{0,2,3,6}.{gamma,weight,bias}  -> ...{norm1,conv1,norm2,conv2}.{...}
    2) Resample:      ...resample.1.{weight,bias}                -> ...conv.{weight,bias}

    不改原 sd，返回新的 OrderedDict，可用于 strict=True。
    """
    out = OrderedDict()

    # ---- 1) residual Sequential -> flat modules ----
    residual_map = {
        0: "norm1",  # RMS_norm(in_dim)
        2: "conv1",  # CausalConv3d(in_dim -> out_dim)
        3: "norm2",  # RMS_norm(out_dim)
        6: "conv2",  # CausalConv3d(out_dim -> out_dim)
        # 1/4/5 (SiLU/SiLU/Dropout) 理论上无参数
    }
    pat_residual = re.compile(r"^(.*)\.residual\.(\d+)\.(weight|bias|gamma)$")

    # ---- 2) resample Sequential conv -> your conv ----
    # checkpoint: encoder.downsamples.2.resample.1.weight
    # your model: encoder.downsamples.2.conv.weight

    pat_resample = re.compile(r"^(.*)\.resample\.1\.(weight|bias)$")

    for k, v in sd.items():
        # resample first (更具体)
        m2 = pat_resample.match(k)
        if m2:
            prefix, param = m2.group(1), m2.group(2)
            new_k = f"{prefix}.conv.{param}"
            out[new_k] = v
            continue

        # residual
        m1 = pat_residual.match(k)
        if m1:
            prefix, idx_str, param = m1.group(1), m1.group(2), m1.group(3)
            idx = int(idx_str)
            if idx in residual_map:
                new_k = f"{prefix}.{residual_map[idx]}.{param}"
                out[new_k] = v
            else:
                # 理论上不会出现（SiLU/Dropout无参数），保留原样以便排查
                out[k] = v
            continue

        # default passthrough
        out[k] = v

    return out

def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        resample_block_size=4,
        residual_main_block_size=4,
        residual_shortcut_block_size=2,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')

    # 用mmap加速
    raw_sd = torch.load(pretrained_path, map_location=device, mmap=True)

    mapped_sd = map_wanvae_ckpt_keys(raw_sd)

    model.load_state_dict(mapped_sd, assign=True)

    print("*" * 40)
    print("Load Model Weight Successfully!!!")
    print("*" * 40)

    # 用mmap加速
    # model.load_state_dict(
    #     torch.load(pretrained_path, map_location=device, mmap=True), assign=True)

    return model

class WanVAE:
    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda",
                 resample_block_size=4,
                 residual_main_block_size=4,
                 residual_shortcut_block_size=2,
                 full_refresh_every=0):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            resample_block_size=resample_block_size,
            residual_main_block_size=residual_main_block_size,
            residual_shortcut_block_size=residual_shortcut_block_size,
            full_refresh_every=full_refresh_every,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(
                    0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
