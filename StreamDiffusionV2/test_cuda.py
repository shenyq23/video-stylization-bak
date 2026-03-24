import torch
import time
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

def benchmark_and_verify():
    # --- 设置 ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 基础参数
    B = 4
    S_dense = 1560
    S_sparse = 156
    H = 12
    D = 128
    
    # KV Cache 参数
    K_max_len = 6 * 1560  # KV Cache 的最大物理长度 (9360)
    K_valid_len = 4*1560    # 当前有效的 KV token 数量
    
    dtype = torch.bfloat16
    device = "cuda"

    print("--- Tensor Initialization ---")
    print(f"Device: {device}, DType: {dtype}")
    print(f"KV Cache Physical Shape: [{B}, {K_max_len}, {H}, {D}]")
    print(f"KV Cache Valid Shape:    [{B}, {K_valid_len}, {H}, {D}]")

    # --- 0. 构造全局 KV Cache 并提取有效部分 ---
    k_cache_full = torch.randn(B, K_max_len, H, D, device=device, dtype=dtype)
    v_cache_full = torch.randn(B, K_max_len, H, D, device=device, dtype=dtype)
    
    # 在实际计算中，我们只需要有效的 KV token
    k_valid = k_cache_full[:, :K_valid_len, :, :].contiguous()
    v_valid = v_cache_full[:, :K_valid_len, :, :].contiguous()

    # --- 1. 4个 chunk 全是稀疏的正常 attn ---
    q_all_sparse = torch.randn(B, S_sparse, H, D, device=device, dtype=dtype)
    
    # --- 2. 4个 chunk 是全量的正常 attn ---
    q_all_dense = torch.randn(B, S_dense, H, D, device=device, dtype=dtype)

    # --- 3 & 4. 2个稀疏 + 2个全量 ---
    # 分离的张量 (用于测试项 4)
    q_dense_half = q_all_dense[:2, :, :, :]   # [2, 1560, H, D]
    q_sparse_half = q_all_sparse[2:, :, :, :] # [2, 156, H, D]
    
    k_valid_dense_half = k_valid[:2, :, :, :]
    v_valid_dense_half = v_valid[:2, :, :, :]
    k_valid_sparse_half = k_valid[2:, :, :, :]
    v_valid_sparse_half = v_valid[2:, :, :, :]

    # 展平的张量 (用于测试项 3)
    # Q 展平: 2个 1560 + 2个 156
    q_packed = torch.cat([
        q_dense_half.reshape(-1, H, D),
        q_sparse_half.reshape(-1, H, D)
    ], dim=0)
    
    # K/V 展平: 4个 1560 (因为 KV 长度都是 1560)
    k_packed = k_valid.reshape(-1, H, D)
    v_packed = v_valid.reshape(-1, H, D)

    # 构造 Varlen 需要的 cu_seqlens
    seq_lens_q = [S_dense, S_dense, S_sparse, S_sparse]
    seq_lens_k = [K_valid_len, K_valid_len, K_valid_len, K_valid_len]
    
    cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens_q), dim=0)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens_k), dim=0)), dtype=torch.int32, device=device)
    
    max_seqlen_q = S_dense
    max_seqlen_k = K_valid_len


    # --- 性能基准测试 ---
    def time_func(func, name, *args, **kwargs):
        # 预热
        for _ in range(5):
            func(*args, **kwargs)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(50):
            func(*args, **kwargs)
        torch.cuda.synchronize()
        ms = (time.time() - start) / 50 * 1000
        print(f"{name:<60} | Time: {ms:.3f} ms")
        return ms

    print("\n--- Benchmark Results ---")
    
    # 测试 1: 4个 chunk 全是稀疏
    time_func(flash_attn_func, 
              f"1. All Sparse Normal Attn (Q:{tuple(q_all_sparse.shape)})", 
              q_all_sparse, k_valid, v_valid)

    # 测试 2: 4个 chunk 全是全量
    time_func(flash_attn_func, 
              f"2. All Dense Normal Attn  (Q:{tuple(q_all_dense.shape)})", 
              q_all_dense, k_valid, v_valid)

    # 测试 3: 2个全量 + 2个稀疏 (Varlen 展平)
    time_func(flash_attn_varlen_func, 
              f"3. Varlen Packed (2 Dense + 2 Sparse, Q:{q_packed.shape[0]} tokens)", 
              q_packed, k_packed, v_packed, 
              cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, 
              max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)

    # 测试 4: 2个全量单独 + 2个稀疏单独
    t_dense_half = time_func(flash_attn_func, 
                             f"4a. Separate Dense Half  (Q:{tuple(q_dense_half.shape)})", 
                             q_dense_half, k_valid_dense_half, v_valid_dense_half)
    
    t_sparse_half = time_func(flash_attn_func, 
                              f"4b. Separate Sparse Half (Q:{tuple(q_sparse_half.shape)})", 
                              q_sparse_half, k_valid_sparse_half, v_valid_sparse_half)
    
    print(f"{'4. Total time for separate calls (4a + 4b):':<60} | Time: {t_dense_half + t_sparse_half:.3f} ms")


    # --- 正确性验证 ---
    print("\n--- Correctness Verification (Test 3 vs Test 4) ---")

    # 计算 Varlen 输出 (Test 3)
    out_packed = flash_attn_varlen_func(
        q_packed, k_packed, v_packed, 
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, 
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k
    )

    # 计算分离调用的输出 (Test 4)
    out_dense_half = flash_attn_func(q_dense_half, k_valid_dense_half, v_valid_dense_half)
    out_sparse_half = flash_attn_func(q_sparse_half, k_valid_sparse_half, v_valid_sparse_half)

    # 将分离计算的结果拼接成一个扁平张量，以对齐 varlen 的输出格式
    out_separate_reconstructed = torch.cat([
        out_dense_half.reshape(-1, H, D),
        out_sparse_half.reshape(-1, H, D)
    ], dim=0)
    
    is_close = torch.allclose(out_packed, out_separate_reconstructed, atol=1e-2, rtol=1e-2)
    print(f"  - Varlen output shape:       {out_packed.shape}")
    print(f"  - Reconstructed output shape:{out_separate_reconstructed.shape}")
    print(f"  - Are they numerically close? -> {is_close}")
    
    if not is_close:
        print("Max diff:", torch.max(torch.abs(out_packed - out_separate_reconstructed)))
    else:
        print("\nVerification passed! Varlen packing computes the exact same result as separate calls.")

if __name__ == "__main__":
    benchmark_and_verify()