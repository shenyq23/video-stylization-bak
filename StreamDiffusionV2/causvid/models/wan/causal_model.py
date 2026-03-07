from causvid.models.wan.wan_base.modules.attention import attention
from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
from flash_attn import flash_attn_interface
import torch.distributed as dist
import time
import numpy as np
import cv2

from gmflow.geometry import flow_warp as universal_flow_warp

def visualize_latent_to_image(latent: torch.Tensor) -> np.ndarray:
    """Visualizes a latent tensor by taking the mean across channels and normalizing."""
    if latent.dim() == 4:
        latent = latent.squeeze(0)

    latent_mean = latent.mean(dim=0)
    min_val, max_val = latent_mean.min(), latent_mean.max()
    if max_val > min_val:
        latent_norm = (latent_mean - min_val) / (max_val - min_val)
    else:
        latent_norm = torch.zeros_like(latent_mean)

    img_np = (latent_norm.float().cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune")

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Vectorized/Parallel implementation of causal_rope_apply.
    """
    # 计时器开始
    # #torch.cuda.synchronize(x.device)
    # start_rope_time = time.time()

    # 1. 获取维度信息
    # B=batch_size, S=sequence_length, N=num_heads, D=head_dim
    B, S, N, D = x.shape
    c = D // 2

    # 2. 切分频率，与原版相同
    # freqs_t: temporal (frame), freqs_h: height, freqs_w: width
    freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 3. 从grid_sizes获取f, h, w
    # 假设批次内所有样本的f, h, w都相同，这在大多数情况下成立
    # 如果不同，需要更复杂的处理，如padding和masking
    f, h, w = grid_sizes[0].tolist()
    assert S == f * h * w, "Sequence length does not match grid dimensions."

    # 4. 并行构建旋转频率 (核心优化点)
    # --------------------------------------------------------------------
    # a. 处理时间维度 (f) 的频率 - 这是因果部分
    # start_frame 是一个 (B,) 的张量，如 [20, 19, 18, 17]
    # 我们需要为每个样本选择从 start_frame[b] 到 start_frame[b] + f - 1 的频率
    # 生成索引: shape will be (B, f)
    # arange(f) -> [0, 1, ..., f-1]
    # start_frame.view(B, 1) -> [[20], [19], [18], [17]]
    # t_indices -> [[20], [19], [18], [17]] (因为 f=1)
    t_indices = torch.arange(f, device=x.device).view(1, f) + start_frame.view(B, 1)

    # 使用高级索引一次性取出所有样本所需的时间频率
    # freqs_t[t_indices] -> shape: (B, f, dim_t)
    causal_freqs_t = freqs_t[t_indices]
    # 变形以支持广播: (B, f, 1, 1, dim_t)
    causal_freqs_t = causal_freqs_t.view(B, f, 1, 1, -1)

    # b. 处理空间维度 (h, w) 的频率 - 非因果部分
    # 这部分对于批次中所有样本都是一样的
    # freqs_h[:h] -> shape: (h, dim_h) -> view to (1, 1, h, 1, dim_h)
    # freqs_w[:w] -> shape: (w, dim_w) -> view to (1, 1, 1, w, dim_w)
    static_freqs_h = freqs_h[:h].view(1, 1, h, 1, -1)
    static_freqs_w = freqs_w[:w].view(1, 1, 1, w, -1)

    # c. 组合所有频率
    # 利用广播机制，将t, h, w的频率组合起来
    # causal_freqs_t (B, f, 1, 1, dim_t)
    # static_freqs_h (1, 1, h, 1, dim_h)
    # static_freqs_w (1, 1, 1, w, dim_w)
    # 它们会自动广播到 (B, f, h, w, dim) 的形状
    # 然后拼接成 (B, f, h, w, c)
    freqs_full = torch.cat([
        causal_freqs_t.expand(B, f, h, w, -1),
        static_freqs_h.expand(B, f, h, w, -1),
        static_freqs_w.expand(B, f, h, w, -1)
    ], dim=-1)

    # d. 变形以匹配输入x的形状
    # (B, f, h, w, c) -> (B, S, c) -> (B, S, 1, c)
    freqs_vectorized = freqs_full.reshape(B, S, c).unsqueeze(2)
    # --------------------------------------------------------------------

    # 5. 并行应用RoPE
    # a. 将x转换为复数形式, (B, S, N, D) -> (B, S, N, c, 2) -> (B, S, N, c)
    x_complex = torch.view_as_complex(x.float().reshape(B, S, N, c, 2))

    # b. 执行旋转: x_complex * freqs_vectorized
    # x_complex shape:      (B, S, N, c)
    # freqs_vectorized shape: (B, S, 1, c)
    # freqs_vectorized 会自动广播到 (B, S, N, c)
    x_rotated = x_complex * freqs_vectorized

    # c. 转换回实数张量, (B, S, N, c) -> (B, S, N, c, 2) -> (B, S, N, D)
    output = torch.view_as_real(x_rotated).flatten(3).type_as(x)

    # (可选) 处理可能的填充。在您的例子中S=f*h*w，所以不需要。
    # 如果 S > f*h*w, 则需要:
    # seq_len = f * h * w
    # output = torch.cat([output[:, :seq_len], x[:, seq_len:]], dim=1)

    # 计时器结束
    # #torch.cuda.synchronize(x.device)
    # end_rope_time = time.time()
    # print(f"### time for PARALLEL apply rope: {end_rope_time - start_rope_time}")

    return output

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.sink_size = 3
        self.adapt_sink_thr = -1

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0
                ,flow_guidance_cache = None, latent_flow_data = None, times_for_rolling = 0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        # print("entering basic self attn:")
        #torch.cuda.synchronize(x.device)
        start_basic_attn_time = time.time()
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        use_flow_guidance = (
            latent_flow_data is not None and
            flow_guidance_cache is not None
        )
        # print(latent_flow_data==None,flow_guidance_cache==None,use_flow_guidance)
        # if (use_flow_guidance):
        #     print(x.shape,flow_guidance_cache.shape, latent_flow_data['flow'].shape,latent_flow_data['mask'].shape)
        #     for i in range (flow_guidance_cache.shape[0]): print(i,torch.mean(flow_guidance_cache[i]),torch.mean(latent_flow_data['flow'][i]),torch.mean(latent_flow_data['mask'][i].float()))

        use_flow_guidance = False
        # #torch.cuda.synchronize(x.device)
        # end_prepare_qkv_time=time.time()
        # print("###time for prepare qkv:",end_prepare_qkv_time-start_basic_attn_time)

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)

        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            num_new_tokens = q.shape[1]
            kv_cache_size = kv_cache["k"].shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            current_start_frame = current_start // frame_seqlen

            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            # This is the same rolling cache logic as before
            is_append_phase = (current_start + num_new_tokens) <= kv_cache_size
            is_rolling_phase = ~is_append_phase
            src_start = sink_tokens + num_new_tokens
            dest_start = sink_tokens
            dest_end = kv_cache_size - num_new_tokens
            write_start = kv_cache_size - num_new_tokens

            roll_start_event = torch.cuda.Event(enable_timing=True)
            roll_end_event = torch.cuda.Event(enable_timing=True)
            roll_start_event.record()

            kv_cache["k"][is_rolling_phase, dest_start:dest_end] = kv_cache["k"][is_rolling_phase, src_start:].clone()
            kv_cache["v"][is_rolling_phase, dest_start:dest_end] = kv_cache["v"][is_rolling_phase, src_start:].clone()

            roll_end_event.record()
            roll_end_event.synchronize()
            roll_time = roll_start_event.elapsed_time(roll_end_event)
            # cnt_true_rolling = is_rolling_phase.sum().item()
            # print(f"rolling cache: {cnt_true_rolling}, time: {roll_time} ms")
            times_for_rolling += roll_time

            kv_cache["k"][is_rolling_phase, write_start:] = roped_key[is_rolling_phase]
            kv_cache["v"][is_rolling_phase, write_start:] = v[is_rolling_phase]
            append_indices = torch.where(is_append_phase)[0]
            if append_indices.numel() > 0:
                append_starts = current_start[append_indices]
                append_offsets = torch.arange(num_new_tokens, device=x.device)
                append_cols = append_starts.unsqueeze(1) + append_offsets.unsqueeze(0)
                append_rows = append_indices.unsqueeze(1).expand(-1, num_new_tokens)
                kv_cache["k"][append_rows, append_cols] = roped_key[append_indices]
                kv_cache["v"][append_rows, append_cols] = v[append_indices]

            effective_seqlens = torch.minimum(
                current_start + num_new_tokens,
                torch.tensor(kv_cache_size, device=current_start.device, dtype=torch.long)
            ).to(torch.int32)

            if use_flow_guidance:
                flow = latent_flow_data["flow"]
                occ_mask = latent_flow_data["mask"] # This is a boolean mask [B, 1, H, W]

                # Reshape previous output x_prev to be image-like for warping
                # x_prev has shape [B, S, C] where S = H*W and C = n*d
                x_prev = flow_guidance_cache
                _, _, latent_h, latent_w = occ_mask.shape
                x_prev_image = x_prev.view(b, latent_h, latent_w, n * d).permute(0, 3, 1, 2)

                x_warped_image = universal_flow_warp(x_prev_image.float(), flow.float()).to(dtype=x.dtype)
                x_warped = x_warped_image.permute(0, 2, 3, 1).view(b, s, n * d)
                x = x_warped.view(b, s, n, d) # Default output is the warped result
                occ_mask_flat = occ_mask.view(b, s) # Shape: [4, 1560]

                num_sparse_tokens = int(torch.sum(occ_mask_flat[0]).item())

                if num_sparse_tokens > 0:
                    # a. Get sparse indices for each batch item using topk. Shape: [B, num_sparse_tokens]
                    _, sparse_indices = torch.topk(occ_mask_flat.byte(), k=num_sparse_tokens, dim=1)

                    # b. Gather sparse queries using the batched indices.
                    # Expand indices from [B, K] to [B, K, N, D] for gather.
                    expanded_indices = sparse_indices.view(b, num_sparse_tokens, 1, 1).expand(-1, -1, n, d)
                    q_sparse = roped_query.gather(1, expanded_indices)

                    x_sparse_output = flash_attn_interface.flash_attn_with_kvcache(
                        q=q_sparse,
                        k_cache=kv_cache["k"],
                        v_cache=kv_cache["v"],
                        cache_seqlens=effective_seqlens,
                    )
                    sparse_results_full = torch.zeros_like(x)
                    sparse_results_full.scatter_(1, expanded_indices, x_sparse_output)
                    occ_mask_expanded = occ_mask_flat.view(b, s, 1, 1).expand_as(x)
                    x = torch.where(occ_mask_expanded, sparse_results_full, x)

            else:
                x = flash_attn_interface.flash_attn_with_kvcache(
                    q=roped_query,
                    k_cache=kv_cache["k"],
                    v_cache=kv_cache["v"],
                    cache_seqlens=effective_seqlens,
                )

        x = x.flatten(2)
        if flow_guidance_cache is not None:
            flow_guidance_cache.copy_(x.detach())
        x = self.o(x)

        #torch.cuda.synchronize(x.device)
        end_basic_attn_time = time.time()
        # print("###time for self attn self.o:",end_basic_attn_time-end_flashattn_time)
        return x, times_for_rolling


class CausalWanAttentionBlock(nn.Module):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                                eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0,
        flow_guidance_cache = None,
        latent_flow_data = None,
        times_for_rolling = 0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        #torch.cuda.synchronize(x.device)
        start_selfattn_block_time=time.time()
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y, times_for_rolling = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, current_end,
            flow_guidance_cache=flow_guidance_cache,
            latent_flow_data=latent_flow_data, times_for_rolling=times_for_rolling)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                 * e[2]).flatten(1, 2)
        #torch.cuda.synchronize(x.device)
        end_selfattn_block_time=time.time()
        # print("###time for self-attn block:",end_selfattn_block_time-start_selfattn_block_time)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        #torch.cuda.synchronize(x.device)
        end_crossattn_ffn_block_time=time.time()
        # print("###time for cross-attn + ffn block:",end_crossattn_ffn_block_time-end_selfattn_block_time)
        return x, times_for_rolling


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1

        self.count=0

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
        block_mode: str = 'input',
        block_num: int = [-1],
        patched_x_shape: torch.Tensor = None,
        flow_guidance_cache = None,
        latent_flow_data = None
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        # print("###entering model:")
        #torch.cuda.synchronize(x.device)
        start_model_time=time.time()
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if block_mode == 'input':
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            bsz, cch, tlen, hh, ww = x[0].shape
            patched_x_shape = torch.tensor([bsz, cch, tlen, hh, ww], dtype=torch.int64, device=device)
        else:
            bsz, cch, tlen, hh, ww = [int(i) for i in patched_x_shape.tolist()]
            x = [u.permute(1,0).reshape(bsz, cch, tlen, hh, ww) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        #torch.cuda.synchronize(x.device)
        # time_embedding_end_time=time.time()
        # print("###time for time embedding:",time_embedding_end_time-start_model_time)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )
        #torch.cuda.synchronize(x.device)
        text_embedding_end_time=time.time()
        start_block_time=text_embedding_end_time
        # print("###time for text embedding:",text_embedding_end_time-time_embedding_end_time)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # print(x.shape,grid_sizes,grid_sizes.shape)
        # x=x[:,156,:]

        # if(grid_sizes[0][0]==1):
        #     for index in range (x.shape[0]):
        #         cur_x=x[index]
        #         cur_size=grid_sizes[index]
        #         cur_x=cur_x.transpose(0,1).view(x.shape[2],cur_size[1],cur_size[2])
        #         visual_x=visualize_latent_to_image(cur_x)
        #         cv2.imwrite(f"./outputs/dit/{self.count}_{index}_pre.png",visual_x)

        times_for_rolling = 0
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                if (block_mode == 'output' or block_mode == 'middle') and block_index < block_num[0]:
                    continue
                if (block_mode == 'input' or block_mode == 'middle') and block_index == block_num[-1]:
                    return x, patched_x_shape
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end,
                        "flow_guidance_cache": None if flow_guidance_cache is None or block_index >= 5 else flow_guidance_cache[block_index],
                        "latent_flow_data": latent_flow_data,
                        "times_for_rolling": times_for_rolling
                    }
                )
                x, times_for_rolling = block(x, **kwargs)
                #torch.cuda.synchronize(x.device)
                block_end_time=time.time()
                # print(f"###time for block {block_index} :",block_end_time-start_block_time)
                start_block_time=block_end_time
        print(f"total time for rolling cache in all blocks: {times_for_rolling:.2f} ms")

        # if(grid_sizes[0][0]==1):
        #     for index in range (x.shape[0]):
        #         cur_x=x[index]
        #         cur_size=grid_sizes[index]
        #         cur_x=cur_x.transpose(0,1).view(x.shape[2],cur_size[1],cur_size[2])
        #         visual_x=visualize_latent_to_image(cur_x)
        #         cv2.imwrite(f"./outputs/dit/{self.count}_{index}_after.png",visual_x)

        # self.count+=1


        if block_mode == 'input' and block_num[-1] == len(self.blocks):
            return x, patched_x_shape

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        #torch.cuda.synchronize(x.device)
        head_end_time=time.time()
        # print("###time for head:",head_end_time-start_block_time)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] *
                x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block
            )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
