import sys
sys.path.append("../StreamDiffusionV2")

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import logging
import threading
from collections import defaultdict # NEW: Import defaultdict
from pynvml import *
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper function to load video (unchanged) ---
def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    if max_frames is not None:
        video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
    return video

# --- GPU占用率监控器 (unchanged) ---
class GPUProfiler:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self._stop_event = threading.Event()
        self._is_profiling_event = threading.Event()
        self.utilizations = []
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.device_id)
            logging.info(f"成功初始化 NVML，监控 GPU {self.device_id}.")
        except NVMLError as error:
            logging.error(f"初始化 NVML 失败: {error}")
            raise
        self.monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
    def _monitor_gpu(self):
        while not self._stop_event.is_set():
            if self._is_profiling_event.is_set():
                try:
                    util = nvmlDeviceGetUtilizationRates(self.handle)
                    self.utilizations.append(util.gpu)
                except NVMLError: pass
            time.sleep(0.01)
    def start(self): self.monitor_thread.start()
    def start_profile(self):
        self.utilizations = []
        self._is_profiling_event.set()
    def stop_profile(self): self._is_profiling_event.clear()
    def get_summary(self):
        if not self.utilizations: return {"avg": 0, "max": 0, "min": 0, "samples": 0}
        return {"avg": sum(self.utilizations) / len(self.utilizations), "max": max(self.utilizations), "min": min(self.utilizations), "samples": len(self.utilizations)}
    def shutdown(self):
        self._stop_event.set()
        self.monitor_thread.join()
        try:
            nvmlShutdown()
            logging.info("NVML 已成功关闭。")
        except NVMLError as error: logging.error(f"关闭 NVML 失败: {error}")

# --- 通用性能分析函数 (unchanged) ---
def profile_module_with_gpu_util(name, func, gpu_profiler, num_iterations=20, warmup=5):
    times = []
    gpu_utils_avg = []
    gpu_utils_max = []
    logging.info(f"正在为 [{name}] 进行预热...")
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()
    logging.info(f"开始分析 [{name}]...")
    for i in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        gpu_profiler.start_profile()
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        gpu_profiler.stop_profile()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)
        summary = gpu_profiler.get_summary()
        gpu_utils_avg.append(summary["avg"])
        gpu_utils_max.append(summary["max"])
        logging.info(f"[{name}] 迭代 {i+1}/{num_iterations}: 耗时: {elapsed_ms:.2f} ms | 平均GPU占用: {summary['avg']:.1f}% | 峰值GPU占用: {summary['max']:.1f}%")
    avg_time = sum(times) / len(times)
    avg_gpu_util = sum(gpu_utils_avg) / len(gpu_utils_avg)
    max_gpu_util = max(gpu_utils_max) if gpu_utils_max else 0
    print("\n" + "="*60)
    print(f"模块 [{name}] 的性能总结")
    print("="*60)
    print(f"在 {num_iterations} 次迭代中的平均耗时: {avg_time:.4f} ms")
    print(f"执行期间的平均GPU占用率: {avg_gpu_util:.2f}%")
    print(f"观测到的峰值GPU占用率: {max_gpu_util:.2f}%")
    print("="*60 + "\n")
    return avg_time, avg_gpu_util, max_gpu_util

# NEW: A dedicated function for detailed profiling
def profile_module_detailed(name, func, num_iterations=20, warmup=5):
    """
    一个用于分析并收集内部模块详细耗时的函数。
    `func` 是一个接受单个参数 `timings` (一个字典) 的可调用对象。
    """
    total_times = []
    detailed_timings_accumulator = defaultdict(list)

    logging.info(f"正在为 [{name}] (详细分析) 进行预热...")
    for _ in range(warmup):
        timings = defaultdict(float) # 在预热时也传递字典，以确保JIT编译等行为一致
        func(timings)
        torch.cuda.synchronize()

    logging.info(f"开始详细分析 [{name}]...")
    for i in range(num_iterations):
        timings = defaultdict(float)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func(timings)
        end_event.record()
        
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        total_times.append(elapsed_ms)
        
        for key, value in timings.items():
            detailed_timings_accumulator[key].append(value)

    avg_total_time = sum(total_times) / len(total_times)
    avg_detailed_times = {key: sum(values) / len(values) for key, values in detailed_timings_accumulator.items()}

    print("\n" + "="*60)
    print(f"模块 [{name}] 的详细性能分解")
    print("="*60)
    print(f"{'总平均耗时':<25}: {avg_total_time:.4f} ms")
    print("-" * 60)
    for key, value in sorted(avg_detailed_times.items()):
        percentage = (value / avg_total_time) * 100 if avg_total_time > 0 else 0
        print(f"{key:<25}: {value:.4f} ms ({percentage:.1f}%)")
    
    manual_sum = sum(avg_detailed_times.values())
    print("-" * 60)
    print(f"{'子模块耗时总和':<25}: {manual_sum:.4f} ms")
    print("="*60 + "\n")

    return avg_total_time, avg_detailed_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Checkpoint folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt_file_path", type=str, required=True, help="Prompt file path")
    parser.add_argument("--video_path", type=str, required=False, default=None, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.700, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--fps_generate", type=int, default=30, help="Target FPS for the producer (VAE encode) thread. Simulates a camera. If 0, runs as fast as possible. Default: 0.")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument("--num_frames", type=int, default=81, help="Video length (number of frames)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device_str = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    
    try:
        device_id = int(device_str.split(':')[-1]) if "cuda" in device_str else 0
    except (ValueError, IndexError):
        device_id = 0

    gpu_profiler = GPUProfiler(device_id=device_id)
    gpu_profiler.start()

    try:
        # --- 1. 加载模型、配置和真实数据 (unchanged) ---
        config = OmegaConf.load(args.config_path)
        config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
        denoising_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0]}
        config.denoising_step_list = denoising_map.get(args.step, [700, 600, 500, 400, 0])

        logging.info("正在初始化 CausalStreamInferencePipeline...")
        pipeline = CausalStreamInferencePipeline(config, device=str(device), text_encoder_on_cpu=True)
        pipeline.to(device=str(device), dtype=dtype)
        
        ckpt_path = os.path.join(args.checkpoint_folder, "model.pt")
        logging.info(f"正在从 {ckpt_path} 加载模型权重")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('generator') or ckpt.get('generator_ema') or ckpt.get('state_dict') or ckpt
        pipeline.generator.load_state_dict(state_dict, strict=False)
        logging.info("模型加载完成。")

        dataset = TextDataset(args.prompt_file_path)
        prompts = [dataset[0]]
        
        ALIGNMENT = 32 
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if new_height != args.height or new_width != args.width:
            logging.warning(f"为对齐调整分辨率: {args.height}x{args.width} -> {new_height}x{new_width}.")
        
        if not args.video_path:
            raise ValueError("--video_path is required for this profiling script.")

        logging.info(f"正在从 {args.video_path} 加载视频...")
        input_video_original = load_mp4_as_tensor(args.video_path, resize_hw=(new_height, new_width)).unsqueeze(0)
        input_video_original = input_video_original.to(device=device, dtype=dtype)
        logging.info(f"输入视频张量形状: {input_video_original.shape}")

        # --- 2. 严格按照原始逻辑，完整地执行一次“冷启动”以准备所有模型状态 (unchanged) ---
        logging.info("正在准备用于分析的真实输入数据和流水线状态...")
        
        chunk_size = 4
        initial_frames_count = 5 
        
        if input_video_original.shape[2] < initial_frames_count + chunk_size:
            raise ValueError(f"视频太短 ({input_video_original.shape[2]} 帧)，需要至少 {initial_frames_count + chunk_size} 帧来进行分析。")

        initial_video_chunk = input_video_original[:, :, 0:initial_frames_count]
        with torch.no_grad():
            latents_for_prepare = pipeline.vae.stream_encode(initial_video_chunk).transpose(2, 1).contiguous()
            noise = torch.randn_like(latents_for_prepare)
            noisy_latents_for_prepare = noise * args.noise_scale + latents_for_prepare * (1 - args.noise_scale)

        prepare_start = 0
        prepare_end = pipeline.frame_seq_length * 2
        with torch.no_grad():
            denoised_pred_for_prepare = pipeline.prepare(
                text_prompts=prompts, device=device, dtype=dtype,
                block_mode='input', noise=noisy_latents_for_prepare, current_start=prepare_start, current_end=prepare_end
            )
        
        with torch.no_grad():
            _ = pipeline.vae.stream_decode_to_pixel(denoised_pred_for_prepare)
        
        torch.cuda.synchronize()
        logging.info("流水线 'prepare' 和 VAE '预热解码' 已完成，所有内部状态均已正确初始化。")
        
        # --- 3. 为“热循环”的各个模块准备输入数据 (unchanged) ---
        inference_video_chunk = input_video_original[:, :, initial_frames_count : initial_frames_count + chunk_size]
        
        with torch.no_grad():
            latents_for_inference = pipeline.vae.stream_encode(inference_video_chunk).transpose(2, 1).contiguous()
            noise = torch.randn_like(latents_for_inference)
            noisy_latents_for_inference = noise * args.noise_scale + latents_for_inference * (1 - args.noise_scale)

        inference_start = prepare_end
        inference_end = inference_start + (chunk_size // 4) * pipeline.frame_seq_length
        with torch.no_grad():
            denoised_latents_from_dit = pipeline.inference_stream(
                noise=noisy_latents_for_inference, 
                current_start=inference_start, 
                current_end=inference_end, 
                current_step=None,
            )
        
        latents_for_decode = denoised_latents_from_dit[[-1]]
        
        logging.info(f"为 VAE 解码准备的 Latent 形状 (热循环模式): {latents_for_decode.shape}")
        logging.info("所有模块的真实输入数据和状态准备就绪。")
        
        # --- 4. 逐个分析选定的计算模块 (unchanged) ---
        
        # 模块 1: VAE Encode
        def vae_encode_op():
            _ = pipeline.vae.stream_encode(inference_video_chunk).transpose(2, 1).contiguous()
        profile_module_with_gpu_util("VAE Encode (Real Data)", vae_encode_op, gpu_profiler)

        # 模块 2: DiT Inference
        def dit_inference_op():
            _ = pipeline.inference_stream(
                noise=noisy_latents_for_inference, 
                current_start=inference_start, 
                current_end=inference_end, 
                current_step=None,
            )
        profile_module_with_gpu_util("DiT Inference (Real Data)", dit_inference_op, gpu_profiler)

        # 模块 3: VAE Decode
        def vae_decode_op():
            _ = pipeline.vae.stream_decode_to_pixel(latents_for_decode)
        profile_module_with_gpu_util("VAE Decode (Real Data)", vae_decode_op, gpu_profiler)

        # --- 5. NEW: 对 VAE 模块进行详细的内部性能分解 ---
        # 我们在这里定义新的操作函数，它们接受一个 `timings` 字典
        
        # 模块 1 详细分析: VAE Encode
        def vae_encode_op_detailed(timings):
            _ = pipeline.vae.stream_encode(inference_video_chunk, timings=timings).transpose(2, 1).contiguous()
        profile_module_detailed("VAE Encode (Detailed)", vae_encode_op_detailed)

        # 模块 3 详细分析: VAE Decode
        def vae_decode_op_detailed(timings):
            _ = pipeline.vae.stream_decode_to_pixel(latents_for_decode, timings=timings)
        profile_module_detailed("VAE Decode (Detailed)", vae_decode_op_detailed)

    finally:
        gpu_profiler.shutdown()


if __name__ == "__main__":
    main()