import torch
import time
import random

def gpu_busy_work(tensor, iterations=100):
    """
    A function that truly keeps the GPU busy by performing multiple matrix multiplications.
    """
    local_tensor = tensor
    for _ in range(iterations):
        local_tensor = torch.matmul(local_tensor, local_tensor)
    return local_tensor

def demonstrate_stream_sync_final():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo.")
        return

    device = torch.device("cuda:0")
    # It's good practice to re-initialize streams for clean experiments
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)

    # --- Scenes 1 and 2 remain the same ---
    print("--- 场景1: 隐式同步 ---")
    data_implicit = torch.ones(1, device=device)
    with torch.cuda.stream(stream1):
        data_implicit.mul_(2)
    with torch.cuda.stream(stream2):
        result_implicit = data_implicit + 1
    torch.cuda.synchronize()
    print(f"隐式同步结果: {result_implicit.item()} (正确, 但依赖隐式行为)\n")

    print("--- 场景2: 使用 Event 进行显式同步 ---")
    data_explicit = torch.ones(1, device=device)
    event = torch.cuda.Event()
    with torch.cuda.stream(stream1):
        data_explicit.mul_(2)
        event.record()
    with torch.cuda.stream(stream2):
        stream2.wait_event(event)
        result_explicit = data_explicit + 1
    torch.cuda.synchronize()
    print(f"显式同步结果: {result_explicit.item()} (正确, 且代码健壮、意图明确)\n")

    # --- 场景3 (最终修正): 隔离竞争操作 ---
    print("--- 场景3 (最终修正): 隔离竞争操作以强制数据竞争 ---")
    print("我们移除了前置的'gpu_busy_work'，只让 add_ 和 mul_ 竞争。")
    
    results = {}
    num_runs = 20 # Increase runs to see more variance
    for i in range(num_runs):
        data_race = torch.ones(1, device=device)
        
        # Enqueue the operations in a tight loop on the CPU side
        # This makes the submission timing itself a factor in the race
        with torch.cuda.stream(stream1):
            data_race.add_(1)

        with torch.cuda.stream(stream2):
            data_race.mul_(3)
        
        # Synchronize *after* both streams have been given their work
        torch.cuda.synchronize()
        result_val = data_race.item()
        results[result_val] = results.get(result_val, 0) + 1
    
    print(f"预期串行结果: 4.0 (mul->add) 或 6.0 (add->mul)")
    print(f"预期并发竞争结果: 2.0 (add wins write) 或 3.0 (mul wins write)")
    print(f"{num_runs}次运行的实际结果分布: {results}")
    print("如果结果中出现了 2.0 或 3.0，就最终证明了数据竞争。\n")

    # --- 场景4 (修正) - Kept for completeness ---
    print("--- 场景4 (修正): 任务的乱序完成 ---")
    data_A = torch.randn(1024, 1024, device=device)
    data_B = torch.randn(256, 256, device=device)
    iterations_A = 20
    iterations_B = 20
    start_event = torch.cuda.Event(enable_timing=True)
    s1_end_event = torch.cuda.Event(enable_timing=True)
    s2_end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() # Warmup and sync
    start_event.record()
    with torch.cuda.stream(stream1):
        stream1.wait_event(start_event)
        gpu_busy_work(data_A, iterations=iterations_A)
        s1_end_event.record()
    with torch.cuda.stream(stream2):
        stream2.wait_event(start_event)
        gpu_busy_work(data_B, iterations=iterations_B)
        s2_end_event.record()
    torch.cuda.synchronize()
    s1_time = start_event.elapsed_time(s1_end_event)
    s2_time = start_event.elapsed_time(s2_end_event)
    print("\n--- 计时结果 ---")
    print(f"Stream 1 ('长'任务) 完成耗时: {s1_time:.2f} ms")
    print(f"Stream 2 ('短'任务) 完成耗时: {s2_time:.2f} ms")
    if s2_time < s1_time:
        print("\n结论: Stream 2 的任务在GPU上先完成了！这清晰地证明了CUDA流的异步和乱序执行能力。")
    else:
        print("\n结论: 在此运行中，Stream 2 并未比 Stream 1 更快完成。")

# 运行最终的演示
demonstrate_stream_sync_final()