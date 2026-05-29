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

from causvid.profile_utils import (
    set_active_timings, clear_active_timings, time_block,
    begin_segment, end_segment,
)

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

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0,pack=None):
    """
    Vectorized/Parallel implementation of causal_rope_apply.
    """
    # 1. 获取维度信息
    # B=batch_size, S=sequence_length, N=num_heads, D=head_dim
    use_sparse = False if pack==None else pack['use_sparse']
    if use_sparse:
        total_tokens, n_heads, d_head = x.shape
        c = d_head // 2
        B = pack['sparse_bs'] + pack['dense_bs']
        
        # 1. 和原来一样，构建完整的频率图
        freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes[0].tolist()
        full_S = f * h * w

        t_indices = torch.arange(f, device=x.device).view(1, f) + start_frame.view(B, 1)
        causal_freqs_t = freqs_t[t_indices].view(B, f, 1, 1, -1)
        static_freqs_h = freqs_h[:h].view(1, 1, h, 1, -1)
        static_freqs_w = freqs_w[:w].view(1, 1, 1, w, -1)

        freqs_full = torch.cat([
            causal_freqs_t.expand(B, f, h, w, -1),
            static_freqs_h.expand(B, f, h, w, -1),
            static_freqs_w.expand(B, f, h, w, -1)
        ], dim=-1)
        freqs_full = freqs_full.reshape(B, full_S, c) # Shape: [4, 1560, C/2]

        # 2. 核心修改：使用索引为扁平化token提取频率
        batch_indices = pack['batch_indices']       # Shape: [3432]
        original_indices = pack['original_indices'] # Shape: [3432]
        
        # freqs_full[batch_indices, original_indices] -> 为每个token找到它在(B,S)矩阵中的位置并取值
        # freqs_gathered = freqs_full[batch_indices, original_indices] # Shape: [3432, C/2]
        flat_indices = batch_indices * full_S + original_indices
        freqs_gathered = freqs_full.view(B * full_S, c)[flat_indices]
        
        # 3. 应用RoPE
        # x shape: [3432, 12, 128] -> reshape to [3432, 12, 64, 2]
        x_complex = torch.view_as_complex(x.float().reshape(total_tokens, n_heads, c, 2))
        
        # freqs_gathered shape: [3432, 64] -> unsqueeze to [3432, 1, 64] for broadcasting
        freqs_vectorized = freqs_gathered.unsqueeze(1)
        
        x_rotated = x_complex * freqs_vectorized # Broadcasting works
        
        output = torch.view_as_real(x_rotated).flatten(2).type_as(x) # Shape: [3432, 12, 128]
        return output
    
    B, S, N, D = x.shape
    c = D // 2

    # 2. 切分频率，与原版相同
    # freqs_t: temporal (frame), freqs_h: height, freqs_w: width
    freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 3. 从grid_sizes获取f, h, w
    # 假设批次内所有样本的f, h, w都相同，这在大多数情况下成立
    # 如果不同，需要更复杂的处理，如padding和masking
    f, h, w = grid_sizes[0].tolist()
    full_S = f * h * w  # 全量序列长度 1560

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


    # print(freqs_t.shape, freqs_h.shape, freqs_w.shape,t_indices.shape, causal_freqs_t.shape, static_freqs_h.shape, static_freqs_w.shape)

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

    freqs_full = freqs_full.reshape(B, full_S, c)
    # d. 变形以匹配输入x的形状
    # (B, f, h, w, c) -> (B, S, c) -> (B, S, 1, c)
    # freqs_vectorized = freqs_full.reshape(B, S, c).unsqueeze(2)
    # --------------------------------------------------------------------

    # 5. 并行应用RoPE
    # a. 将x转换为复数形式, (B, S, N, D) -> (B, S, N, c, 2) -> (B, S, N, c)
    x_complex = torch.view_as_complex(x.float().reshape(B, S, N, c, 2))


    # b. 执行旋转: x_complex * freqs_vectorized
    # x_complex shape:      (B, S, N, c)
    # freqs_vectorized shape: (B, S, 1, c)
    # freqs_vectorized 会自动广播到 (B, S, N, c)
    # if mask is not None:
    #     # mask 形状为 (B, full_S)
    #     mask_flat = mask.view(B, full_S)
        
    #     # 使用和外部 gather x 时完全相同的 topk 逻辑获取 sparse_indices
    #     # 这里的 S 就是稀疏后的 token 数量 (156)
    #     _, sparse_indices = torch.topk(mask_flat.byte(), k=S, dim=1)
        
    #     # 扩展 indices 以便 gather freqs_full: (B, S, c)
    #     expanded_indices = sparse_indices.unsqueeze(-1).expand(-1, -1, c)
        
    #     # 提取稀疏 token 对应的频率: (B, 1560, c) -> (B, 156, c)
    #     freqs_full = freqs_full.gather(1, expanded_indices)
    # ===============================================

    # 变形以匹配输入x的形状: (B, S, 1, c)
    freqs_vectorized = freqs_full.unsqueeze(2)

    # 5. 并行应用RoPE
    x_rotated = x_complex * freqs_vectorized
    # print(x_rotated.shape)

    # print(freqs_full.shape, freqs_vectorized.shape, x_complex.shape,x_rotated.shape)

    # c. 转换回实数张量, (B, S, N, c) -> (B, S, N, c, 2) -> (B, S, N, D)
    output = torch.view_as_real(x_rotated).flatten(3).type_as(x)

    # (可选) 处理可能的填充。在您的例子中S=f*h*w，所以不需要。
    # 如果 S > f*h*w, 则需要:
    # seq_len = f * h * w
    # output = torch.cat([output[:, :seq_len], x[:, seq_len:]], dim=1)

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

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0, pack = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        use_sparse = False if pack==None else pack['use_sparse']
        mask=pack['mask'] if 'mask' in pack else None

        if use_sparse:
            total_tokens, C = x.shape
            n, d = self.num_heads, self.head_dim

            # 1. QKV linear projections
            with time_block("DiT/Linear"):
                q = self.norm_q(self.q(x))
                k = self.norm_k(self.k(x))
                v = self.v(x)
                q = q.view(total_tokens, n, d)
                k = k.view(total_tokens, n, d)
                v = v.view(total_tokens, n, d)

            # 2. RoPE
            current_start_frame = current_start // pack['frame_seqlen']
            with time_block("DiT/RoPE"):
                roped_query = causal_rope_apply(q, grid_sizes, freqs, current_start_frame, pack).type_as(v)
                roped_key = causal_rope_apply(k, grid_sizes, freqs, current_start_frame, pack).type_as(v)

            # 3. KV cache write + packing for varlen flash attention.
            # Bucketed as "Warp" because in the paper this is the cost of
            # maintaining the inter-chunk cache that flow-warp consumes.
            B = pack['sparse_bs'] + pack['dense_bs']
            batch_indices = pack['batch_indices']
            local_indices = pack['original_indices']
            write_start = pack['write_start']
            with time_block("DiT/Warp"):
                start_offsets = write_start[batch_indices]
                sequence_indices = local_indices + start_offsets
                kv_cache["k"].index_put_((batch_indices, sequence_indices), roped_key)
                kv_cache["v"].index_put_((batch_indices, sequence_indices), v)

                effective_seqlens = pack['effective_seqlens']
                is_full = pack['is_full']
                if is_full:
                    k_packed = kv_cache["k"].reshape(-1, n, d)
                    v_packed = kv_cache["v"].reshape(-1, n, d)
                else:
                    k_valid_list = []
                    v_valid_list = []
                    for i in range(B):
                        seq_len_i = effective_seqlens[i].item()
                        k_valid_list.append(kv_cache["k"][i, :seq_len_i, :, :])
                        v_valid_list.append(kv_cache["v"][i, :seq_len_i, :, :])
                    k_packed = torch.cat(k_valid_list, dim=0)
                    v_packed = torch.cat(v_valid_list, dim=0)

                cu_seqlens_k = pack['cu_seqlens_k']
                max_seqlen_k = pack['max_seqlen_k']
                max_seqlen_q = pack['max_seqlen_q']

            # 4. Self attention kernel
            with time_block("DiT/Self Attn"):
                x = flash_attn_interface.flash_attn_varlen_func(
                    q=roped_query,
                    k=k_packed,
                    v=v_packed,
                    cu_seqlens_q=pack['cu_seqlens'],
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )

            # 5. Output linear
            with time_block("DiT/Linear"):
                x = x.flatten(1)
                x = self.o(x)
            return x

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

        
        # print(latent_flow_data==None,flow_guidance_cache==None,use_flow_guidance)
        # if (use_flow_guidance):
        #     print(x.shape,flow_guidance_cache.shape, latent_flow_data['flow'].shape,latent_flow_data['mask'].shape)
        #     for i in range (flow_guidance_cache.shape[0]): print(i,torch.mean(flow_guidance_cache[i]),torch.mean(latent_flow_data['flow'][i]),torch.mean(latent_flow_data['mask'][i].float()))

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
            frame_seqlen = x.shape[1] // grid_sizes[0, 0].item()
            num_new_tokens = q.shape[1]
            kv_cache_size = kv_cache["k"].shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            current_start_frame = current_start // frame_seqlen

            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            # 1. 计算环形缓冲区的容量
            ring_capacity = kv_cache_size - sink_tokens

            # 2. 计算每个 Batch 样本的写入起始位置 (O(1) 计算，无需数据搬运)
            # 如果还没满 (current_start < kv_cache_size)，就按顺序写
            # 如果满了，就在 [sink_tokens, kv_cache_size) 之间循环覆盖最老的数据
            write_starts = torch.where(
                current_start < kv_cache_size,
                current_start,
                sink_tokens + ((current_start - kv_cache_size) % ring_capacity)
            )

            # 3. 向量化写入 Cache (高级索引)
            B = x.shape[0]
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_new_tokens)
            offsets = torch.arange(num_new_tokens, device=x.device).unsqueeze(0)
            write_cols = write_starts.unsqueeze(1) + offsets

            # 直接把新 Token 覆盖到算好的位置，零拷贝！
            kv_cache["k"][batch_indices, write_cols] = roped_key
            kv_cache["v"][batch_indices, write_cols] = v

            # 4. 计算有效长度 (和原来保持一致)
            effective_seqlens = torch.minimum(
                current_start + num_new_tokens,
                torch.tensor(kv_cache_size, device=current_start.device, dtype=torch.long)
            ).to(torch.int32)

            if use_sparse:
                pass

            else:
                x = flash_attn_interface.flash_attn_with_kvcache(
                    q=roped_query,
                    k_cache=kv_cache["k"],
                    v_cache=kv_cache["v"],
                    cache_seqlens=effective_seqlens,
                )

        x = x.flatten(2)
        x = self.o(x)

        #torch.cuda.synchronize(x.device)
        end_basic_attn_time = time.time()
        # print("###time for self attn self.o:",end_basic_attn_time-end_flashattn_time)
        return x


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
        pack = None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        use_sparse = False if pack==None else pack['use_sparse']
        if use_sparse:
            batch_indices = pack['batch_indices']

            # 1. Modulation chunks + norm1 + pre-attn affine (all linear-ish)
            with time_block("DiT/Linear"):
                e_chunks = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
                e0_gathered = e_chunks[0][batch_indices].squeeze(1).squeeze(1)
                e1_gathered = e_chunks[1][batch_indices].squeeze(1).squeeze(1)
                norm1_out = self.norm1(x)
                attn_x = norm1_out * (1 + e1_gathered) + e0_gathered

            # 2. Self-attention (self.self_attn is internally instrumented into
            # Linear/RoPE/Warp/Self Attn buckets — no wrap here, would double count)
            y = self.self_attn(
                attn_x,
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, current_end,
                pack=pack)

            # 3. Post-attn residual + cross-attn (whole call is the bucket)
            with time_block("DiT/Linear"):
                e2_gathered = e_chunks[2][batch_indices].squeeze(1).squeeze(1)
                x = x + y * e2_gathered

            with time_block("DiT/Cross Attn"):
                cross_attn_out = self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache, pack=pack)
                x = x + cross_attn_out

            # 4. FFN modulation + FFN + post-FFN residual
            with time_block("DiT/Linear"):
                e3_gathered = e_chunks[3][batch_indices].squeeze(1).squeeze(1)
                e4_gathered = e_chunks[4][batch_indices].squeeze(1).squeeze(1)
                norm2_out = self.norm2(x)
                ffn_in = norm2_out * (1 + e4_gathered) + e3_gathered
                ffn_out = self.ffn(ffn_in)
                e5_gathered = e_chunks[5][batch_indices].squeeze(1).squeeze(1)
                x = x + ffn_out * e5_gathered

            return x
        #torch.cuda.synchronize(x.device)
        start_selfattn_block_time=time.time()
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        # print(x.shape,len(e),e[0].shape)
        attn_x=self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        attn_x=attn_x * (1 + e[1]) + e[0]
        # print(attn_x.shape)
        attn_x=attn_x.flatten(1, 2)
        y = self.self_attn(
            attn_x,
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, current_end,
            pack=pack)

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
        return x


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
        x=self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        # print(e[0].shape,x.shape)
        x = (self.head(x*(1 + e[1]) + e[0]))
        return x
    
    def forward_sparse(self, x, e,pack):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        batch_indices = pack['batch_indices']
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        e_head_0_gathered = e[0][batch_indices].squeeze(1).squeeze(1)
        e_head_1_gathered = e[1][batch_indices].squeeze(1).squeeze(1)
        
        norm_out = self.norm(x) # x shape: [3432, 1536]
        head_in = norm_out * (1 + e_head_1_gathered) + e_head_0_gathered
        x = self.head(head_in)
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
        self.flow_guidance_cache=None

        # Per-submodule profiling: when set to a dict by gpu.py, `time_block`
        # calls inside forward() accumulate into it. None disables profiling.
        self.profile_timings = None

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

        use_sparse=(latent_flow_data!=None and self.flow_guidance_cache!=None) and self.count%5!=0
        pack={}
        pack['use_sparse'] = use_sparse

        # print("before dit for:",x.shape)

        # torch.cuda.synchronize(x.device)
        # start=time.time()

        # ===================== 1. 将输入拆分为 Sparse 和 Dense 两部分 =====================
        # The whole pack-setup block is the "Warp" overhead in MotionFlow — building
        # gather indices, cu_seqlens, write_starts etc. that the cache warp relies on.
        _dit_warp_setup = begin_segment("DiT/Warp") if use_sparse else None
        if use_sparse:
            B, S, C = x.shape
            # Half the streaming-batch chunks (the early high-noise ones) are sparse:
            # B=4 (4-step) -> 2, B=2 (2-step) -> 1. Matches paper B_s.
            sparse_bs = B // 2
            dense_bs = B - sparse_bs
            # --- a. 拆分数据和掩码 ---
            x_sparse_full = x[:sparse_bs]
            x_dense = x[sparse_bs:]
            
            occ_mask_half = latent_flow_data["mask_half"].view(B, S)
            occ_mask_sparse = occ_mask_half[:sparse_bs]

            num_sparse_tokens = 0
            x_sparse_gathered = torch.empty(sparse_bs, 0, C, device=x.device, dtype=x.dtype)
            sparse_indices = torch.empty(sparse_bs, 0, device=x.device, dtype=torch.long)

            # 只有在存在稀疏样本时才计算 num_sparse_tokens 并收集数据
            if sparse_bs > 0:
                # 假设所有稀疏样本都减少到相同数量的令牌
                num_sparse_tokens = int(torch.sum(occ_mask_sparse[0]).item())
                
                if num_sparse_tokens > 0:
                    # 获取稀疏令牌的索引，形状为 (sparse_bs, num_sparse_tokens)
                    _, sparse_indices = torch.topk(occ_mask_sparse.byte(), k=num_sparse_tokens, dim=1)
                    # 扩展索引以便在最后一维收集数据
                    expanded_indices = sparse_indices.unsqueeze(-1).expand(-1, -1, C)
                    # 从原始稀疏张量中收集令牌
                    x_sparse_gathered = x_sparse_full.gather(1, expanded_indices)

            x_flat = torch.cat([
                x_sparse_gathered.view(-1, C), # 如果 sparse_bs=0, shape is (0, C)
                x_dense.view(-1, C)            # 如果 dense_bs=0, shape is (0, C)
            ], dim=0)
            
            batch_ids = torch.arange(B, device=x.device)
            # `lengths` 张量的构建可以正确处理 sparse_bs 或 dense_bs 为 0 的情况。
            # torch.full((0,), ...) 会创建一个空张量，torch.cat 会忽略它。
            lengths = torch.cat([
                torch.full((sparse_bs,), num_sparse_tokens, device=x.device, dtype=torch.long),
                torch.full((dense_bs,), S, device=x.device, dtype=torch.long)
            ])
            # `repeat_interleave` 也能正确处理空的 `lengths` 或 `repeats`。
            batch_indices = torch.repeat_interleave(batch_ids, repeats=lengths)

            # ii. `original_indices`: 每个令牌在其原始序列中的位置。
            dense_indices_template = torch.arange(S, device=x.device)
            # .expand(0, -1) 会创建一个 shape 为 (0, S) 的空张量，这是正确的。
            dense_indices = dense_indices_template.expand(dense_bs, -1)
            # 再次利用 torch.cat 对空张量的处理能力。

            # print(sparse_indices.shape,dense_indices.shape)
            original_indices = torch.cat([
                sparse_indices.reshape(-1),
                dense_indices.reshape(-1)
            ], dim=0)

            # iii. `cu_seqlens`: 累积序列长度。
            lengths_with_zero = torch.cat([torch.tensor([0], device=x.device, dtype=torch.long), lengths])
            cu_seqlens = torch.cumsum(lengths_with_zero, dim=0,dtype=torch.int32)

            #self attention pre compute
            kv_cache_size = kv_cache[0]["k"].shape[1]
            full_seq_len = S

            effective_seqlens = torch.minimum(
                current_start + full_seq_len, 
                torch.tensor(kv_cache_size, device=x.device, dtype=torch.long)
            ).to(torch.int32)
            
            # 预计算 max_seqlen (使用 GPU 上的 max，不调用 .item())
            pack['max_seqlen_q'] = S 
            pack['max_seqlen_k'] = effective_seqlens.max() # 保持为 Tensor
            
            # 构造 cu_seqlens_k
            pack['cu_seqlens_k'] = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.cumsum(effective_seqlens, dim=0, dtype=torch.int32)
            ])
            
            # 存入 effective_seqlens 用于后续拼接
            pack['effective_seqlens'] = effective_seqlens
            pack['is_full']=(effective_seqlens == kv_cache_size).all()

            frame_seqlen = S // grid_sizes[0, 0].item()
            pack['frame_seqlen']=frame_seqlen
            sink_tokens = self.blocks[0].self_attn.sink_size * frame_seqlen
            ring_capacity = kv_cache_size - sink_tokens
            pack['write_start']=torch.where(
                current_start < kv_cache_size,
                current_start,
                sink_tokens + ((current_start - kv_cache_size) % ring_capacity)
            )



            #T2V cross-attn 需要的 context 长度,pre compute
            b = context.size(0)
            if context_lens is None:
                context_lens = torch.tensor([crossattn_cache[0]["k"].shape[1]] * b, device=crossattn_cache[0]["k"].device, dtype=torch.int32)
            pack['cu_seqlens_ctx'] = torch.cat([context_lens.new_zeros([1]), context_lens.cumsum(0, dtype=torch.int32)])
            pack['max_seqlen_ctx'] = context_lens.max() # 保持为 Tensor



            # --- e. 更新 kwargs 和输入张量以供后续模块使用 ---
            pack.update({
                'batch_indices': batch_indices,
                'original_indices': original_indices,
                'cu_seqlens': cu_seqlens,
                'lengths': lengths,
                'sparse_bs': sparse_bs,
                'dense_bs': dense_bs,
                'num_sparse_tokens': num_sparse_tokens,
                'full_seq_len': S,
                'sparse_indices': sparse_indices, # 用于最后解包
                'mask': occ_mask_sparse # 传递给RoPE
            })
            x = x_flat
        end_segment(_dit_warp_setup)

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
                        "pack": pack,
                    }
                )
                # torch.cuda.synchronize(x.device)
                # end=time.time()
                x = block(x, **kwargs)
                # torch.cuda.synchronize(x.device)
                # block_end_time=time.time()
                # print(f"###time for block {block_index} :",block_end_time-end)
                # start_block_time=block_end_time
        
        if block_mode == 'input' and block_num[-1] == len(self.blocks):
            return x, patched_x_shape

        # ===================== 3. Head 处理与重新拼接 =====================
        e_head = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)

        # print("after dit",x.shape,e_head.shape)

        # torch.cuda.synchronize(x.device)
        # start=time.time()

        # Head call (forward_sparse is a Linear+norm op) plus the scatter/unpack
        # that reassembles per-sample dense tensors from packed sparse tokens.
        # We split: head -> Linear bucket, scatter/unpack -> Warp bucket.
        if use_sparse:
            with time_block("DiT/Linear"):
                x = self.head.forward_sparse(x, e_head, pack)
            _dit_warp_unpack = begin_segment("DiT/Warp")
            sparse_bs = pack['sparse_bs']
            dense_bs = pack['dense_bs']
            num_sparse_tokens = pack['num_sparse_tokens']
            full_seq_len = pack['full_seq_len']
            out_channels = x.shape[-1]
            B = sparse_bs + dense_bs

            x_sparse_full = torch.zeros(sparse_bs, full_seq_len, out_channels, device=x.device, dtype=x.dtype)
            if sparse_bs > 0:
                num_total_sparse = sparse_bs * num_sparse_tokens
                x_sparse_packed = x[:num_total_sparse].view(sparse_bs, num_sparse_tokens, out_channels)
                expanded_indices = pack['sparse_indices'].unsqueeze(-1).expand(-1, -1, out_channels)
                x_sparse_full.scatter_(
                    dim=1,
                    index=expanded_indices,
                    src=x_sparse_packed,
                )

            x_dense_full = torch.empty(dense_bs, full_seq_len, out_channels, device=x.device, dtype=x.dtype)
            if dense_bs > 0:
                num_total_sparse = sparse_bs * num_sparse_tokens
                x_dense_packed = x[num_total_sparse:]
                x_dense_full = x_dense_packed.view(dense_bs, full_seq_len, out_channels)

            x = torch.cat([x_sparse_full, x_dense_full], dim=0)
            x = x.unsqueeze(1)
            end_segment(_dit_warp_unpack)
        else:
            with time_block("DiT/Linear"):
                x = self.head(x, e_head)

        # print("head",x.shape)

        # torch.cuda.synchronize(x.device)
        # end=time.time()
        # print("unpack",end-start) #0.00058
        

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        # print(x.shape)

        # torch.cuda.synchronize(x.device)
        # start=time.time()

        # ===================== 4. 光流 Warp 引导 (仅作用于前两个 chunk) =====================
        if use_sparse and sparse_bs>0:
            with time_block("DiT/Warp"):
                b, s, _, h, w = x.shape
                x = x.squeeze(2)
                flow = latent_flow_data["flow"][:sparse_bs]
                occ_mask = latent_flow_data["mask"][:sparse_bs]
                x_prev = self.flow_guidance_cache[:sparse_bs]
                x_prev_image = x_prev.squeeze(2)

                x_warped = universal_flow_warp(x_prev_image.float(), flow.float()).to(dtype=x.dtype)
                occ_mask_expanded = occ_mask.view(sparse_bs, 1, h, w).expand(sparse_bs, s, h, w)
                x_final = torch.where(occ_mask_expanded, x[:sparse_bs], x_warped)
                x[:sparse_bs] = x_final
                x = x.unsqueeze(2)
            
        if latent_flow_data is not None: 
            self.flow_guidance_cache = x
        self.count+=1

        # cur_k=kv_cache[0]['k']
        # print("cur_k",cur_k.shape,x.shape)
        # b,s,n,d=cur_k.shape
        # cur_k=cur_k.view(b,s,n*d)
        # cur_k=torch.mean(cur_k,dim=-1)
        # total_chunk=3
        # token_per_chunk=s//total_chunk
        # h=x.shape[-2]//2
        # w=x.shape[-1]//2
        # for i in range (b):
        #     for j in range (total_chunk):
        #         cur_view=cur_k[i, j*token_per_chunk:(j+1)*token_per_chunk]
        #         cur_view=cur_view.view(h,w)
        #         print("batch token",i,j,cur_view.shape)
        #         for coor_x in range(h):
        #             print(coor_x,cur_view[coor_x,:].mean())




        # torch.cuda.synchronize(x.device)
        # end=time.time()

        # print("warp",end-start) #4.95e-5

        return x

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
        set_active_timings(self.profile_timings)
        try:
            if kwargs.get('kv_cache', None) is not None:
                return self._forward_inference(*args, **kwargs)
            else:
                return self._forward_train(*args, **kwargs)
        finally:
            clear_active_timings()

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
