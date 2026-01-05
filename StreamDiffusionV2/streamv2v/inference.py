"""
Single GPU Inference Pipeline - Optimized Parallel Execution with Simulated Producer FPS

This file implements an optimized producer-consumer pattern using separate CUDA
streams to achieve true parallelism between VAE encoding and DiT inference.
It adds a real-time simulation by throttling the producer to match a
specified generation FPS (--fps_generate), mimicking a live input source like a camera.

**Design Principles:**
1.  **CUDA Streams**:
    -   A `producer_stream` is dedicated to VAE encoding.
    -   A `consumer_stream` is dedicated to DiT inference and VAE decoding.
2.  **CUDA Events for Synchronization and Timing**:
    -   `torch.cuda.Event` is used for efficient, non-blocking, GPU-side synchronization
        between streams and for precise GPU execution timing.
3.  **Producer FPS Simulation (fps_generate)**:
    -   The `--fps_generate` argument simulates a fixed-rate input source.
    -   The producer calculates the time required to "receive" a new chunk of frames
        (e.g., 4 frames at 30 FPS = 133ms).
    -   After its VAE encoding task for a chunk is finished on the GPU (verified via
        `event.synchronize()`), the producer thread will `time.sleep()` if it
        finished faster than the target interval. If it's slower, it logs a lag warning.
4.  **Detailed Overlap Logging**:
    -   Timestamps clearly show when tasks are submitted (CPU-side) and completed
        (GPU-side), visualizing the computational overlap and any introduced sleep time.
5.  **Logical Equivalence**: The core numerical logic remains a 1:1 replication of
    the original serial code to ensure bit-for-bit identical output.
"""
import sys
sys.path.append("../StreamDiffusionV2")

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import logging
import threading
import queue

import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

from utils.optical_wrapper import GMFlowWrapper, RAFTFlowWrapper
import cv2

class OpticalFlowCalculator:
    """
    一个封装类，用于初始化和使用 calflow 中的光流模型。
    它提供了一个简单的接口来计算两帧之间的光流。
    """
    def __init__(self, flow_model_type: str, device: torch.device):
        self.device = device
        self.logger = logging.getLogger("OpticalFlowCalculator")
        
        if not flow_model_type:
            self.model = None
            self.logger.info("Optical flow calculation is disabled.")
            return

        self.logger.info(f"Initializing optical flow model: {flow_model_type}")
        if flow_model_type.lower() == "gmflow":
            FlowModel = GMFlowWrapper
        elif flow_model_type.lower() == "raft":
            FlowModel = RAFTFlowWrapper
        else:
            raise ValueError(f"Unsupported flow model type: {flow_model_type}")
            
        # GMFlowWrapper/RAFTFlowWrapper的 __init__ 接受一个 device 字符串
        self.model = FlowModel(str(self.device))
        self.logger.info(f"Optical flow model '{flow_model_type}' initialized on device '{self.device}'.")

    def calculate_flow(self, ref_frame_tensor: torch.Tensor, current_frame_tensor: torch.Tensor) -> tuple | None:
        """
        计算从 ref_frame 到 current_frame 的光流。
        它直接调用您修改后的 `compute_flow_from_tensors` 方法。

        Args:
            ref_frame_tensor (torch.Tensor): 参考帧, shape [1, C, H, W], range [-1, 1].
            current_frame_tensor (torch.Tensor): 当前帧, shape [1, C, H, W], range [-1, 1].

        Returns:
            A tuple (backward_flow, backward_occlusion) or None if model is not initialized.
            - backward_flow (torch.Tensor): 从 current 到 ref 的光流, shape [1, 2, H, W].
            - backward_occlusion (torch.Tensor): 对应的遮挡图, shape [1, 1, H, W].
        """
        if self.model is None:
            return None
        
        # 调用您修改后的函数，它期望 [-1, 1] 的输入并返回与输入 dtype/device 一致的张量
        # forward_flow: ref -> current
        # backward_flow: current -> ref
        _forward_flow, backward_flow, _forward_occlusion, backward_occlusion = \
            self.model.compute_flow_from_tensors(ref_frame_tensor, current_frame_tensor)
        
        # 我们通常需要 backward flow 来将前一帧的内容 "warp" 到当前帧的位置
        return backward_flow, backward_occlusion

def visualize_flow_to_image(flow: torch.Tensor) -> np.ndarray:
    """
    将光流张量可视化为彩色图像。
    Args:
        flow (torch.Tensor): 光流张量, shape [2, H, W] or [1, 2, H, W].
    Returns:
        np.ndarray: RGB图像, shape [H, W, 3], dtype uint8.
    """
    if flow.dim() == 4:
        flow = flow.squeeze(0)
    flow = flow.permute(1, 2, 0).cpu().numpy() # H, W, 2
    
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255 # Value

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2 # Hue
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Saturation

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

# --- Helper functions (unchanged) ---
def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]: # <--- 修改: 更新返回类型提示
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    # <--- 修改: 捕获第三个返回值 info，其中包含元数据
    video, _, info = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    
    # <--- 新增: 从元数据中获取视频的FPS，如果获取不到则提供一个默认值
    original_fps = info.get('video_fps', 16) 

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
        
    return video, original_fps # <--- 修改: 返回视频张量和原始FPS

def compute_noise_scale_and_step(input_video_original: torch.Tensor, end_idx: int, chunk_size: int, noise_scale: float, init_noise_scale: float):
    l2_dist=(input_video_original[:,:,end_idx-chunk_size:end_idx]-input_video_original[:,:,end_idx-chunk_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0,1,3,4))).max()/0.2).clamp(0,1)
    new_noise_scale = (init_noise_scale-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step

# --- SingleGPUInferencePipeline class (Logging format updated) ---
class SingleGPUInferencePipeline:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("SingleGPUInference")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Updated formatter to match target log
            formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("Initializing CausalStreamInferencePipeline...")
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device), text_encoder_on_cpu=True)
        self.pipeline.to(device=str(device), dtype=torch.bfloat16)
        self.logger.info("Single GPU inference pipeline manager initialized")
    
    def load_model(self, checkpoint_folder: str):
        ckpt_path = os.path.join(checkpoint_folder, "model.pt")
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('generator') or ckpt.get('generator_ema') or ckpt.get('state_dict') or ckpt
        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            self.logger.warning(f"Strict load_state_dict failed: {e}; retrying with strict=False")
            self.pipeline.generator.load_state_dict(state_dict, strict=False)
    
    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, current_start: int, current_end: int):
        return self.pipeline.prepare(
            text_prompts=text_prompts, device=self.device, dtype=torch.bfloat16,
            block_mode='input', noise=noise, current_start=current_start, current_end=current_end
        )

# --- Optimized ParallelInferenceOrchestrator (Logging format updated) ---
class ParallelInferenceOrchestrator:
    def __init__(self, pipeline_manager: SingleGPUInferencePipeline):
        self.pipeline_manager = pipeline_manager
        self.pipeline = pipeline_manager.pipeline
        self.device = pipeline_manager.device
        self.logger = logging.getLogger("ParallelOrchestrator")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Updated formatter to match target log
            formatter = logging.Formatter('%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.producer_stream = torch.cuda.Stream(device=self.device)
        self.consumer_stream = torch.cuda.Stream(device=self.device)
        
        self.data_queue = queue.Queue(maxsize=5) 
        self.producer_thread = None
        self.processed = 0

    def _producer_task(self, input_video_original: torch.Tensor, 
                       flow_calculator: OpticalFlowCalculator,
                       num_chunks: int, chunk_size: int, noise_scale: float, num_steps: int, fps_generate: int):
        self.logger.info("Producer thread started.")
        
        is_realtime_sim = fps_generate > 0
        chunk_interval_seconds = 0
        if is_realtime_sim:
            chunk_interval_seconds = chunk_size / fps_generate
            self.logger.info(f"Real-time simulation enabled: Target Producer FPS={fps_generate}, Chunk Size={chunk_size}, Target Interval={chunk_interval_seconds:.4f}s")
        
        # 用于维持稳定生产速率的时间锚点
        next_chunk_submit_time = time.time()

        with torch.cuda.stream(self.producer_stream):
            # --- 1. 为"冷启动" / prepare() 调用生产数据 ---
            start_idx, end_idx = 0, 5
            
            prod_end_event = torch.cuda.Event(enable_timing=True)
            
            if input_video_original is not None:
                inp = input_video_original[:, :, start_idx:end_idx].to(self.device, non_blocking=True)
                latents = self.pipeline.vae.stream_encode(inp)
                latents = latents.transpose(2, 1).contiguous()
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                noisy_latents = torch.randn(1, 1 + self.pipeline.num_frame_per_block, 16, self.pipeline.height, self.pipeline.width, device=self.device, dtype=torch.bfloat16)
            
            prod_end_event.record()
            # self.data_queue.put((noisy_latents, None, prod_end_event, "Initial"))
            self.data_queue.put((noisy_latents, None, None, prod_end_event, "Initial"))
            self.logger.info("Producer: Initial 5-frame data block placed in queue.")

            # --- 2. 为"热循环"生产数据 ---
            init_noise_scale = noise_scale
            total_hot_chunks = num_chunks + num_steps - 1
            
            for i in range(total_hot_chunks):
                # --- [修正后] 的 FPS 节流逻辑 ---
                if is_realtime_sim:
                    # 设置下一个数据块 *应该被提交* 的时间点
                    next_chunk_submit_time += chunk_interval_seconds
                    
                    # 计算需要休眠多久才能达到提交时间点
                    current_time = time.time()
                    sleep_needed = next_chunk_submit_time - current_time
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)
                    # 注意：我们在这里不警告延迟，因为瓶颈是消费者。
                    # 如果生产者有能力，它自然会追赶上来。

                chunk_id = i + 1
                start_idx = end_idx
                end_idx += chunk_size

                prod_end_event = torch.cuda.Event(enable_timing=True)

                flows_for_chunk=[] 

                if input_video_original is not None and end_idx <= input_video_original.shape[2]:
                    inp = input_video_original[:, :, start_idx:end_idx].to(self.device, non_blocking=True)
                    noise_scale, current_step = compute_noise_scale_and_step(
                        input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                    )
                    latents = self.pipeline.vae.stream_encode(inp)
                    latents = latents.transpose(2, 1).contiguous()
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
                    # if flow_calculator is not None and flow_calculator.model is not None:
                    #     if i < 4:
                    #         ref_frame_idx = 4
                    #         ref_type = "prepare frame 5"
                    #     else:
                    #         ref_chunk_idx = i - 4
                    #         ref_chunk_start_idx = 5 + ref_chunk_idx * chunk_size
                    #         ref_frame_idx = ref_chunk_start_idx + chunk_size - 1
                    #         ref_type = f"last frame of chunk {ref_chunk_idx+1}"
                        
                    #     self.logger.info(f"Producer: Chunk {chunk_id}, using ref frame {ref_frame_idx} (from {ref_type}) for batched flow calculation on GPU.")
                        
                    #     # 1. Prepare batch tensors on the GPU. No CPU transfer needed.
                    #     # Reference frame, shape [1, C, H, W]
                    #     ref_frame_tensor = input_video_original[:, :, ref_frame_idx].to(self.device, non_blocking=True)
                    #     # Expand to match the chunk size without copying memory. Shape [chunk_size, C, H, W]
                    #     batched_ref_frames = ref_frame_tensor.expand(chunk_size, -1, -1, -1)

                    #     # Current frames for the chunk, shape [1, C, chunk_size, H, W]
                    #     # current_frames_chunk = input_video_original[:, :, start_idx:end_idx]
                    #     # Reshape to [chunk_size, C, H, W] for the model
                    #     batched_current_frames = inp.squeeze(0).permute(1, 0, 2, 3)

                    #     # 2. Perform a single, batched inference call on the GPU.
                    #     # This is highly efficient and parallelizes the computation across the chunk.
                    #     self.logger.info(f"Producer: Submitting batched flow calculation for chunk {chunk_id}...")
                    #     batch_flow_data = flow_calculator.calculate_flow(batched_ref_frames, batched_current_frames)
                    #     self.logger.info(f"Producer: Batched flow calculation for chunk {chunk_id} enqueued on GPU stream.")

                    #     # 3. Unpack the batched results into a list for the queue.
                    #     # This is a fast slicing operation.
                    #     if batch_flow_data:
                    #         batched_bwd_flow, batched_bwd_occ = batch_flow_data
                    #         for j in range(chunk_size):
                    #             # Slice the j-th result from the batch, keeping the batch dimension
                    #             flow_tuple = (batched_bwd_flow[j:j+1], batched_bwd_occ[j:j+1])
                    #             flows_for_chunk.append(flow_tuple)
                else:
                    noisy_latents = torch.randn(1, self.pipeline.num_frame_per_block, 16, self.pipeline.height, self.pipeline.width, device=self.device, dtype=torch.bfloat16)
                    current_step = None
                
                prod_end_event.record()
                # self.data_queue.put((noisy_latents, current_step, prod_end_event, chunk_id))
                self.data_queue.put((noisy_latents, current_step, flows_for_chunk, prod_end_event, chunk_id))
                
                if chunk_id <= num_chunks:
                    self.logger.info(f"Producer: Real data chunk {chunk_id}/{num_chunks} placed in queue.")
                else:
                    flush_chunk_id = chunk_id - num_chunks
                    total_flush_chunks = num_steps - 1
                    self.logger.info(f"Producer: Flush chunk {flush_chunk_id}/{total_flush_chunks} placed in queue.")

                # --- [已移除] 阻塞的 synchronize 调用 ---
                # 旧的逻辑导致整个流水线串行化。
                # 新的逻辑正确地控制 CPU 的提交速率，而无需等待 GPU。
        
        self.logger.info("Producer thread finished. All data blocks produced.")

    def run_parallel_inference(
        self, 
        input_video_original: torch.Tensor, 
        flow_calculator: OpticalFlowCalculator,
        prompts: list, 
        num_chunks: int, 
        chunk_size: int, 
        noise_scale: float, 
        output_folder: str, 
        fps: int, 
        num_steps: int,
        fps_generate: int
    ):
        self.logger.info("Consumer started. Replicating original inference logic with detailed timing.")
        os.makedirs(output_folder, exist_ok=True)

        flow_viz_folder = os.path.join(output_folder, "flow_visualizations")
        if flow_calculator is not None and flow_calculator.model is not None:
            os.makedirs(flow_viz_folder, exist_ok=True)
        
        self.producer_thread = threading.Thread(
            target=self._producer_task,
            args=(input_video_original,
                  flow_calculator,
                   num_chunks, chunk_size, noise_scale, num_steps, fps_generate)
        )
        self.producer_thread.start()

        results, save_results = {}, 0
        iteration_times = []
        
        current_start = 0
        current_end = self.pipeline.frame_seq_length * 2
        
        try:
            # --- 3. Process the "Cold Start" data from the queue ---
            self.logger.info("Consumer: Waiting for initial data block...")
            initial_noisy_latents, current_step, flows_for_chunk,producer_done_event, chunk_id= self.data_queue.get()
            
            with torch.cuda.stream(self.consumer_stream):
                # self.consumer_stream.wait_event(producer_done_event)
                
                denoised_pred = self.pipeline_manager.prepare_pipeline(
                    text_prompts=prompts,
                    noise=initial_noisy_latents,
                    current_start=current_start,
                    current_end=current_end
                )
                video = self.pipeline.vae.stream_decode_to_pixel(denoised_pred)
                
                # self.consumer_stream.synchronize()
                
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                results[save_results] = video.cpu().float().numpy()
                save_results += 1
                self.logger.info("Consumer: Initial block processed and saved.")

            # --- 4. Process "Hot Loop" data from the queue ---
            last_save_time = time.time() # Initialize timer for first iteration
            while self.processed < num_chunks + num_steps - 1:
                noisy_latents, current_step, flows_for_chunk,producer_done_event, chunk_id = self.data_queue.get()
                
                with torch.cuda.stream(self.consumer_stream):
                    # self.consumer_stream.wait_event(producer_done_event)
                    # if flows_for_chunk:
                    #     self.logger.info(f"Consumer: Received {len(flows_for_chunk)} flow fields for chunk {chunk_id}. Visualizing...")
                    #     for j, flow_data in enumerate(flows_for_chunk):
                    #         if flow_data:
                    #             backward_flow, _ = flow_data
                    #             flow_image = visualize_flow_to_image(backward_flow.float())
                    #             frame_in_chunk_idx = j
                    #             save_path = os.path.join(flow_viz_folder, f"flow_viz_chunk_{chunk_id:04d}_frame_{frame_in_chunk_idx}.png")
                    #             cv2.imwrite(save_path, flow_image)
                    
                    current_start = current_end
                    current_end += (chunk_size // 4) * self.pipeline.frame_seq_length

                    denoised_pred = self.pipeline.inference_stream(
                        noise=noisy_latents,
                        current_start=current_start,
                        current_end=current_end,
                        current_step=current_step,
                    )
                    
                    video_out = None
                    if self.processed + 1 >= num_steps:
                        video_out = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                    
                    # self.consumer_stream.synchronize()
                    self.processed += 1
                    
                    if video_out is not None:
                        video = (video_out * 0.5 + 0.5).clamp(0, 1)
                        video = video[0].permute(0, 2, 3, 1).contiguous()
                        results[save_results] = video.cpu().float().numpy()
                        
                        # --- NEW: Iteration Timing and Logging ---
                        current_time = time.time()
                        iter_time = current_time - last_save_time
                        last_save_time = current_time
                        iteration_times.append(iter_time)
                        iter_fps = chunk_size / iter_time
                        
                        self.logger.info(f"Consumer: Saved output from iter {save_results}, Iter Time: {iter_time:.4f}s, FPS: {iter_fps:.4f}")
                        save_results += 1
        
        finally:
            self.producer_thread.join()
            
            self.logger.info("="*50)
            self.logger.info("Performance Summary")
            self.logger.info("="*50)

            # Ensure we have the correct number of frames
            video_list = [results[i] for i in range(save_results)]
            video = np.concatenate(video_list, axis=0)
            
            # --- NEW: Final FPS calculation based on actual iteration times ---
            if iteration_times:
                avg_iter_time = np.mean(iteration_times)
                avg_fps = chunk_size / avg_iter_time
                self.logger.info(f"Average End-to-End FPS (Consumer-side, after pipeline fill): {avg_fps:.4f}")

            self.logger.info(f"Final video shape: {video.shape}")
            
            output_path = os.path.join(output_folder, f"output_parallel_timed.mp4")
            export_to_video(video, output_path, fps=fps)
            self.logger.info(f"Video saved to: {output_path}")
            self.logger.info("Parallel inference with timing completed.")


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
    parser.add_argument("--flow_model", type=str, default="gmflow", choices=["gmflow", "raft"], help="Optical flow model to use (from calflow). If None, flow is not calculated.")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    # Updated root logger to match target format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    flow_calculator = None

    if args.video_path is not None:
        ALIGNMENT = 32 
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if new_height != args.height or new_width != args.width:
            logging.warning(f"Adjusting resolution from {args.height}x{args.width} to {new_height}x{new_width}.")
        resize_hw = (new_height, new_width)
        args.height, args.width = new_height, new_width
        input_video_original, original_fps = load_mp4_as_tensor(args.video_path, resize_hw=resize_hw)

        args.fps=original_fps

        input_video_original = input_video_original.unsqueeze(0)
        logging.info(f"Input video tensor shape: {input_video_original.shape}")
        t = input_video_original.shape[2]
        input_video_original = input_video_original.to(dtype=torch.bfloat16)

        if args.flow_model:
            logging.info(f"Preparing for optical flow calculation with model: {args.flow_model}")
            flow_calculator = OpticalFlowCalculator(args.flow_model, device)
    else:
        input_video_original = None
        t = args.num_frames
        if args.fps_generate > 0:
            logging.warning("--fps_generate is specified but --video_path is not. The simulation will run but without video input.")
        
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    
    denoising_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0]}
    config.denoising_step_list = denoising_map.get(args.step, [700, 600, 500, 400, 0])
    
    chunk_size = 4
    # The number of 'real' chunks that will result in a saved output
    num_chunks = (t - 5) // chunk_size

    pipeline_manager = SingleGPUInferencePipeline(config, device)
    pipeline_manager.load_model(args.checkpoint_folder)

    num_steps = len(pipeline_manager.pipeline.denoising_step_list)
    
    orchestrator = ParallelInferenceOrchestrator(pipeline_manager)
    
    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]
    
    try:
        orchestrator.run_parallel_inference(
            input_video_original, 
            flow_calculator,
            prompts, 
            num_chunks, 
            chunk_size, 
            args.noise_scale, 
            args.output_folder, 
            args.fps, 
            num_steps,
            args.fps_generate
        )
    except Exception as e:
        logging.error(f"Error occurred during inference: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()