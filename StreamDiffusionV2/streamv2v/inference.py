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
sys.path.append("../")
sys.path.append("../deps/gmflow")

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
import torch.nn.functional as F

from utils.optical_wrapper import GMFlowWrapper, RAFTFlowWrapper, OcclusionComputation, X265MVWrapper
import cv2
from gmflow.geometry import flow_warp as universal_flow_warp
import json

class OpticalFlowCalculator:
    def __init__(self, 
                 flow_model_type: str, 
                 device: torch.device, 
                 x265_params: dict = None,
                 occlusion_method: str = 'quantile',
                 top_k_percentage: float = 0.1,
                 morph_kernel_size: int = 7,
                 conn_comp_threshold_quantile: float = 0.75
                ):
        self.device = device
        self.logger = logging.getLogger("OpticalFlowCalculator")
        self.x265_params = x265_params or {}
        self.flow_model_type = flow_model_type
        self.occlusion_method = occlusion_method
        self.top_k_percentage=top_k_percentage
        self.morph_kernel_size = morph_kernel_size
        self.conn_comp_threshold_quantile = conn_comp_threshold_quantile
        
        self.logger.info(f"Using occlusion mask generation method: '{self.occlusion_method}'")

        if not flow_model_type or flow_model_type.lower() == 'none':
            self.model = None
            self.logger.info("Optical flow calculation is disabled.")
            return

        self.logger.info(f"Initializing optical flow model: {flow_model_type}")
        FlowModel = {"gmflow": GMFlowWrapper, "raft": RAFTFlowWrapper, "x265": X265MVWrapper}.get(flow_model_type.lower())
        if FlowModel is None:
            raise ValueError(f"Unsupported flow model type: {flow_model_type}")

        if flow_model_type=="x265": self.model = FlowModel(str(self.device),native_x265=True)
        else: self.model = FlowModel(str(self.device))
        
        if self.flow_model_type.lower() == 'x265':
             self.logger.info("Using 'luminosity' occlusion for x265.")
             self.occlusion_computer = OcclusionComputation(use_luminosity=True)
        else:
             self.logger.info("Using 'geometry' occlusion for DL models.")
             self.occlusion_computer = OcclusionComputation(use_geometry=True)

    def compute_binary_occlusion_mask(self, raw_occ_map: torch.Tensor) -> torch.Tensor:
        B, _, H, W = raw_occ_map.shape
        final_masks = []

        for i in range(B):
            single_occ_map = raw_occ_map[i, 0]

            if self.occlusion_method == 'exact':
                num_elements = single_occ_map.numel()
                k = int(num_elements * self.top_k_percentage)
                
                # 确保 k 至少为 1 (如果百分比 > 0)，且不超过总元素数
                k = max(1, min(k, num_elements)) if self.top_k_percentage > 0 else 0

                if k == 0:
                    binary_mask = torch.zeros_like(single_occ_map, dtype=torch.bool)
                else:
                    # 展平张量并找到前 k 大的值的索引
                    flat_map = single_occ_map.flatten()
                    _, topk_indices = torch.topk(flat_map, k)
                    
                    # 创建一个新的布尔掩码，并将 topk 索引位置设为 True
                    binary_mask_flat = torch.zeros_like(flat_map, dtype=torch.bool)
                    binary_mask_flat.scatter_(0, topk_indices, True)
                    
                    # 恢复为原始的 2D 形状
                    binary_mask = binary_mask_flat.view(H, W)

            elif self.occlusion_method == 'quantile':
                threshold = torch.quantile(single_occ_map, 1.0 - self.top_k_percentage)
                binary_mask = (single_occ_map >= threshold)
            elif self.occlusion_method == 'morphological':
                initial_quantile = max(0.5, 1.0 - self.top_k_percentage * 2)
                threshold = torch.quantile(single_occ_map, initial_quantile)
                noisy_mask_np = (single_occ_map > threshold).cpu().numpy().astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
                cleaned_mask_np = cv2.morphologyEx(noisy_mask_np, cv2.MORPH_OPEN, kernel)
                binary_mask = torch.from_numpy(cleaned_mask_np).to(self.device, dtype=torch.bool)
            elif self.occlusion_method == 'connected_components':
                threshold = torch.quantile(single_occ_map, self.conn_comp_threshold_quantile)
                binary_mask_np = (single_occ_map > threshold).cpu().numpy().astype(np.uint8)
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)

                if num_labels <= 1:
                    final_masks.append(torch.zeros_like(single_occ_map, dtype=torch.bool))
                    continue

                region_scores = []
                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area < self.morph_kernel_size * self.morph_kernel_size: 
                        continue
                    region_mask_np = (labels_im == label)
                    score = single_occ_map[torch.from_numpy(region_mask_np).to(self.device)].mean().item()
                    region_scores.append({'id': label, 'score': score, 'area': area})

                region_scores.sort(key=lambda x: x['score'], reverse=True)
                final_mask_np = np.zeros((H, W), dtype=bool)
                target_area = H * W * self.top_k_percentage
                covered_area = 0
                for region in region_scores:
                    if covered_area >= target_area: break
                    region_mask_np = (labels_im == region['id'])
                    final_mask_np[region_mask_np] = True
                    covered_area += region['area']
                
                binary_mask = torch.from_numpy(final_mask_np).to(self.device)
            else:
                raise ValueError(f"Unsupported occlusion method: {self.occlusion_method}")

            final_masks.append(binary_mask)

        return torch.stack(final_masks, dim=0).unsqueeze(1)


    def calculate_flow(self, ref_frame: torch.Tensor, current_frame: torch.Tensor) -> tuple | None:
        if self.model is None: return None
        
        if (self.flow_model_type=="x265"): 
            # print(self.x265_params)
            fwd_flow, bwd_flow = self.model.compute_flow_from_tensors(ref_frame, current_frame, **self.x265_params)
        else:
            fwd_flow, bwd_flow = self.model.compute_flow_from_tensors(ref_frame, current_frame)
        
        fwd_occ, bwd_occ = self.occlusion_computer(ref_frame, current_frame, fwd_flow, bwd_flow)

        if bwd_occ.dim() == 3:
            bwd_occ = bwd_occ.unsqueeze(1)
            fwd_occ = fwd_occ.unsqueeze(1)

        return bwd_flow, bwd_occ

def tensor_to_np_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts a [-1, 1] or [0, 1] image tensor to a [0, 255] uint8 RGB numpy array."""
    if tensor.min() < -0.1:
        tensor = (tensor * 0.5 + 0.5)
    tensor = tensor.clamp(0, 1)

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    np_img = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    return (np_img * 255).astype(np.uint8)

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

def visualize_flow_to_rgb(flow: torch.Tensor, vector_stride: int = 20) -> np.ndarray:
    """
    Visualizes an optical flow tensor by drawing arrows on a black background.
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError("Input flow must be a [B, 2, H, W] tensor.")

    B, _, H, W = flow.shape
    flow_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arrow_color = (0, 255, 0) # Green in BGR

    for y in range(vector_stride // 2, H, vector_stride):
        for x in range(vector_stride // 2, W, vector_stride):
            dx, dy = flow_np[y, x, :]
            start_point = (x, y)
            end_x = int(np.clip(round(x + dx), 0, W - 1))
            end_y = int(np.clip(round(y + dy), 0, H - 1))
            end_point = (end_x, end_y)
            cv2.arrowedLine(flow_canvas, start_point, end_point, arrow_color, 1, tipLength=0.3)
            
    return cv2.cvtColor(flow_canvas, cv2.COLOR_BGR2RGB)

def overlay_flow_on_image(image: np.ndarray, flow_viz: np.ndarray) -> np.ndarray:
    """
    Overlays the flow visualization (arrows) on top of a background image.
    
    Args:
        image (np.ndarray): The background RGB image (H, W, 3).
        flow_viz (np.ndarray): The flow visualization RGB image (H, W, 3) with arrows.
        
    Returns:
        np.ndarray: The combined image.
    """
    if image.shape != flow_viz.shape:
        h, w, _ = image.shape
        flow_viz = cv2.resize(flow_viz, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # cv2.add performs saturated addition, which is perfect for this overlay effect.
    # It adds the green arrow color to the background image pixels.
    overlayed_image = cv2.add(image, cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR))
    return overlayed_image
    return cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

def visualize_flow_with_source_overlay(
    source_image: np.ndarray, 
    target_image: np.ndarray, 
    flow_viz: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """
    创建一个复合视觉效果：在目标图像上叠加光流箭头，然后再与半透明的源图像进行混合。

    这个视觉效果旨在取代简单的光流箭头可视化，以提供更丰富的上下文。
    最终图像的计算方式为: (target_image + flow_arrows) * (1-alpha) + source_image * alpha.
    
    Args:
        source_image (np.ndarray): 用于半透明叠加的源 RGB 图像 (H, W, 3)。
        target_image (np.ndarray): 作为背景的目标 RGB 图像 (H, W, 3)。
        flow_viz (np.ndarray): 在黑色背景上带有光流箭头的 RGB 视觉效果图 (H, W, 3)。
        alpha (float): 源图像叠加的透明度/权重。
        
    Returns:
        np.ndarray: 合成后的 RGB 图像。
    """
    # 确保所有输入图像的尺寸与目标图像一致
    h, w, _ = target_image.shape
    if source_image.shape[:2] != (h, w):
        source_image = cv2.resize(source_image, (w, h), interpolation=cv2.INTER_AREA)
    if flow_viz.shape[:2] != (h, w):
        flow_viz = cv2.resize(flow_viz, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- OpenCV 操作需要 BGR 格式 ---
    source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    flow_viz_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)

    # 步骤 1: 创建基础叠加层 (目标图像 + 光流箭头)
    # cv2.add 执行饱和加法，非常适合添加绿色箭头。
    base_overlay_bgr = cv2.add(target_bgr, flow_viz_bgr)

    # 步骤 2: 在基础叠加层之上混合半透明的源图像
    # 公式为: dst = src1*alpha + src2*(1-alpha) + gamma
    composite_bgr = cv2.addWeighted(source_bgr, alpha, base_overlay_bgr, 1.0 - alpha, 0.0)
    
    # --- 将最终图像转换回 RGB 格式，以与其他可视化函数保持一致 ---
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
    
    return composite_rgb

def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
    device: str = 'cuda:0',
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

        video = video.to(device) if torch.cuda.is_available() and str(device).startswith('cuda') else video
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
    def __init__(self, config, device: torch.device, use_cached_text_embedding: bool = False):
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
        # <--- MODIFIED LINE: Pass the new argument to the pipeline --->
        self.pipeline = CausalStreamInferencePipeline(config, device=str(device), text_encoder_on_cpu=True, use_cached_text_embedding=use_cached_text_embedding)
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
        self.save_queue = queue.Queue()
        self.producer_thread = None
        self.saver_thread = None
        self.processed = 0
        self.prev_latent=None

    def _producer_task(self, input_video_original: torch.Tensor, 
                       flow_calculator: OpticalFlowCalculator,
                       num_chunks: int, chunk_size: int, noise_scale: float, num_steps: int, fps_generate: int):
        self.logger.info("Producer thread started.")

        # torch.cuda.synchronize(device=self.device)
        # mem_producer_start = torch.cuda.memory_reserved(device=self.device)
        
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
            self.logger.info(f"Producer: Initial 5-frame data block placed in queue.{time.time()-next_chunk_submit_time}")

            # --- 2. 为"热循环"生产数据 ---
            init_noise_scale = noise_scale
            total_hot_chunks = num_chunks + num_steps - 1

            self.prev_latent=None

            # torch.cuda.synchronize(device=self.device)
            # mem_prepare_end = torch.cuda.memory_reserved(device=self.device)
            # print("GPU memory used by producer during prepare(): ", (mem_prepare_end - mem_producer_start)/1024/1024/1024, "GB","from",mem_producer_start/1024/1024/1024,"GB to",mem_prepare_end/1024/1024/1024,"GB")
            
            for i in range(total_hot_chunks):
                # --- [修正后] 的 FPS 节流逻辑 ---
                if is_realtime_sim:
                    # 设置下一个数据块 *应该被提交* 的时间点
                    next_chunk_submit_time += chunk_interval_seconds
                    
                    # 计算需要休眠多久才能达到提交时间点
                    current_time = time.time()
                    sleep_needed = next_chunk_submit_time - current_time
                    if sleep_needed > 0:
                        self.logger.warning(f"Producer is sleeping for {sleep_needed:.4f}s")
                        time.sleep(sleep_needed)
                    else:
                        self.logger.warning(f"Producer is lagging behind real-time schedule by {-sleep_needed:.4f}s")

                chunk_id = i + 1
                start_idx = end_idx
                end_idx += chunk_size
                prod_end_event = torch.cuda.Event(enable_timing=True)
                flow_data=None
                if input_video_original is not None and end_idx <= input_video_original.shape[2]:
                    inp = input_video_original[:, :, start_idx:end_idx].to(self.device, non_blocking=True)
                    noise_scale, current_step = compute_noise_scale_and_step(
                        input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale
                    )
                    latents = self.pipeline.vae.stream_encode(inp)
                    latents = latents.transpose(2, 1).contiguous()
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
                    # print(flow_calculator.device)
                    if flow_calculator is not None and chunk_id>1:
                        ref_frame_idx = start_idx -1
                        target_frame_idx = end_idx-1
                        ref_frame_tensor = input_video_original[:, :, ref_frame_idx].to(device=self.device,dtype=torch.float32, non_blocking=True)
                        target_frame_tensor = input_video_original[:, :, target_frame_idx].to(self.device,dtype=torch.float32, non_blocking=True)
                        # self.logger.info(f"prepare float data {time.time()-current_time}")
                        self.logger.info(f"Producer: Submitting batched flow calculation for chunk {chunk_id}...")
                        flow_data = flow_calculator.calculate_flow(ref_frame_tensor,target_frame_tensor)
                        self.logger.info(f"Producer: Batched flow calculation for chunk {chunk_id} enqueued on GPU stream.")
                    
                        # bwd_flow,bwd_occ=flow_data
                        # print(bwd_flow.shape,bwd_occ.shape,bwd_flow.dtype,bwd_occ.dtype)
                        # bwd_flow=torch.ones((1,2,480,832)).to(dtype=torch.float32,device=ref_frame_tensor.device)
                        # bwd_occ=torch.ones((1,1,480,832)).to(dtype=torch.float32,device=ref_frame_tensor.device)
                        # flow_data=bwd_flow,bwd_occ
                else:
                    noisy_latents = torch.randn(1, self.pipeline.num_frame_per_block, 16, self.pipeline.height, self.pipeline.width, device=self.device, dtype=torch.bfloat16)
                    current_step = None
                
                ##visualize
                
                
                if (flow_data!=None): 
                    bwd_flow,bwd_occ=flow_data
                    # warped_frame = universal_flow_warp(ref_frame_tensor, bwd_flow)
                    # pixel_binary_mask = flow_calculator.compute_binary_occlusion_mask(bwd_occ)
                    # occluded_warped_frame = torch.where(
                    #     pixel_binary_mask.to(device=warped_frame.device), 
                    #     torch.zeros_like(warped_frame), 
                    #     warped_frame
                    # )
                    # viz_frame_n = tensor_to_np_img(ref_frame_tensor)
                    # viz_frame_n1 = tensor_to_np_img(target_frame_tensor)
                    # # viz_warped_frame = tensor_to_np_img(occluded_warped_frame)
                    # viz_warped_frame = tensor_to_np_img(warped_frame)
                    # viz_frame_flow = visualize_flow_to_rgb(bwd_flow, vector_stride=20)
                    # viz_frame_compo=visualize_flow_with_source_overlay(viz_frame_n, viz_frame_n1, viz_frame_flow, alpha=0.4)
                    # pixel_row = np.concatenate([viz_frame_n, viz_frame_n1, viz_warped_frame, viz_frame_compo], axis=1)
                    # final_image= pixel_row
                    # save_path = os.path.join("./outputs/warped", f"comparison_chunk_{chunk_id:04d}.png")
                    # cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

                    latent_n_repr = self.prev_latent.squeeze(1)
                    latent_n1_repr = latents.squeeze(1)
                    _, _, latent_h, latent_w = latent_n_repr.shape

                    downsampled_flow = F.interpolate(bwd_flow, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
                    downsampled_flow *= (float(latent_h) / bwd_flow.shape[2])
                    warped_latent = universal_flow_warp(latent_n_repr.float(), downsampled_flow.float())
                    downsampled_occ = F.interpolate(bwd_occ, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
                    latent_binary_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)
                    fused_latent_repr = torch.where(latent_binary_mask, latent_n1_repr, warped_latent).to(dtype=latents.dtype)
                    latents=fused_latent_repr
                    # print(i,torch.mean((latents-latent_n1_repr)**2).item())
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

                    target_h=latent_h//2
                    target_w=latent_w//2
                    downsampled_flow = F.interpolate(bwd_flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    downsampled_flow *= (float(target_h) / bwd_flow.shape[2])
                    downsampled_occ= F.interpolate(bwd_occ, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    latent_binary_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)
                    flow_data= (downsampled_flow,latent_binary_mask)
                    # print("in producer",i, torch.mean(downsampled_flow),torch.mean(latent_binary_mask.float()))
                    # flow_data=None
                


                prod_end_event.record()
                self.prev_latent=latents
                self.data_queue.put((noisy_latents, current_step, flow_data, prod_end_event, chunk_id))
                
                if chunk_id <= num_chunks:
                    self.logger.info(f"Producer: Real data chunk {chunk_id}/{num_chunks} placed in queue. ")
                else:
                    flush_chunk_id = chunk_id - num_chunks
                    total_flush_chunks = num_steps - 1
                    self.logger.info(f"Producer: Flush chunk {flush_chunk_id}/{total_flush_chunks} placed in queue. ")
        
        self.logger.info("Producer thread finished. All data blocks produced.")

    def _saver_task(self, results_dict: dict):
        self.logger.info("Saver thread started.")
        last_save_time=time.time()
        chunk_size=4
        iteration_times = []
        while True:
            # Get data from the save queue
            item = self.save_queue.get()
            
            # Sentinel value to signal termination
            if item is None:
                self.logger.info("Saver thread received termination signal.")
                break
            
            cpu_tensor_future, index = item
            
            # This line will block THIS (saver) thread until the specific
            # non-blocking transfer initiated by the consumer is complete.
            # The main consumer thread is NOT blocked.
            numpy_array = cpu_tensor_future.float().numpy()
            
            results_dict[index] = numpy_array
            self.logger.debug(f"Saver: Saved numpy array for index {index}.")
            current_time = time.time()
            iter_time = current_time - last_save_time
            last_save_time = current_time
            iteration_times.append(iter_time)
            iter_fps = chunk_size / iter_time
            self.logger.info(f"Saver: Render Video Chunk for iter {index}, Iter Time: {iter_time:.4f}s, FPS: {iter_fps:.4f}")
        
        if iteration_times:
            iteration_times=np.array(iteration_times)
            iteration_times=iteration_times[1:]
            avg_iter_time = np.mean(iteration_times)
            avg_fps = chunk_size / avg_iter_time
            self.logger.info(f"Average End-to-End FPS (Saver-side, after pipeline fill): {avg_fps:.4f}")
        self.logger.info("Saver thread finished.")

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
        # torch.cuda.synchronize(device=self.device)
        # mem_run_start = torch.cuda.memory_reserved(device=self.device)


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

        results = {}
        self.saver_thread = threading.Thread(target=self._saver_task, args=(results,))
        self.saver_thread.start()

        # results, save_results = {}, 0
        iteration_times = []
        save_results=0
        
        current_start = 0
        current_end = self.pipeline.frame_seq_length * 2
        
        try:
            # --- 3. Process the "Cold Start" data from the queue ---
            self.logger.info("Consumer: Waiting for initial data block...")
            initial_noisy_latents, current_step, flows_for_chunk,producer_done_event, chunk_id= self.data_queue.get()
            
            with torch.cuda.stream(self.consumer_stream):
                self.consumer_stream.wait_event(producer_done_event)

                self.logger.info(f"Consumer: Got initial data block. ")
                
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
                self.save_queue.put((video.to('cpu', non_blocking=True), save_results))
                save_results += 1
                self.logger.info("Consumer: Initial block processed and enqueued for saving.")
                
                # video = (video * 0.5 + 0.5).clamp(0, 1)
                # video = video[0].permute(0, 2, 3, 1).contiguous()
                # results[save_results] = video.cpu().float().numpy()
                # save_results += 1
                # self.logger.info("Consumer: Initial block processed and saved.")
            
            # torch.cuda.synchronize(device=self.device)
            # mem_run_end = torch.cuda.memory_reserved(device=self.device)
            # print("GPU memory used by consumer during run(): ", (mem_run_end - mem_run_start)/1024/1024/1024, "GB","from",mem_run_start/1024/1024/1024,"GB to",mem_run_end/1024/1024/1024,"GB")
            # --- 4. Process "Hot Loop" data from the queue ---
            last_save_time = time.time() # Initialize timer for first iteration
            while self.processed < num_chunks + num_steps - 1:
                noisy_latents, current_step, flows_for_chunk,producer_done_event, chunk_id = self.data_queue.get()
                
                with torch.cuda.stream(self.consumer_stream): 
                    self.consumer_stream.wait_event(producer_done_event)
                    self.logger.info(f"Consumer: Got data block {self.processed+1}.")
                    current_start = current_end
                    current_end += (chunk_size // 4) * self.pipeline.frame_seq_length

                    denoised_pred = self.pipeline.inference_stream(
                        noise=noisy_latents,
                        current_start=current_start,
                        current_end=current_end,
                        current_step=current_step,
                        latent_flow_data=flows_for_chunk,
                    )
                    
                    video_out = None
                    if self.processed + 1 >= num_steps:
                        video_out = self.pipeline.vae.stream_decode_to_pixel(denoised_pred[[-1]])
                    
                    # self.consumer_stream.synchronize()
                    self.processed += 1
                    
                    if video_out is not None:
                        video = (video_out * 0.5 + 0.5).clamp(0, 1)
                        video = video[0].permute(0, 2, 3, 1).contiguous()
                        self.save_queue.put((video.to('cpu', non_blocking=True), save_results))

                        # video = (video_out * 0.5 + 0.5).clamp(0, 1)
                        # video = video[0].permute(0, 2, 3, 1).contiguous()
                        # results[save_results] = video.cpu().float().numpy()
                        
                        # --- NEW: Iteration Timing and Logging ---
                        current_time = time.time()
                        iter_time = current_time - last_save_time
                        last_save_time = current_time
                        iteration_times.append(iter_time)
                        iter_fps = chunk_size / iter_time
                        
                        self.logger.info(f"Consumer: Enqueued output for iter {save_results}, Iter Time: {iter_time:.4f}s, FPS: {iter_fps:.4f}")
                        save_results += 1

                    # torch.cuda.synchronize(device=self.device)
                    # mem_run_end2 = torch.cuda.memory_reserved(device=self.device)
                    # print(self.processed,"GPU memory used by consumer during run(): ", (mem_run_end2 - mem_run_end)/1024/1024/1024, "GB","from",mem_run_end/1024/1024/1024,"GB to",mem_run_end2/1024/1024/1024,"GB")
        
        finally:
            self.producer_thread.join()

            self.save_queue.put(None) # Sentinel value
            self.saver_thread.join()
            
            self.logger.info("="*50)
            self.logger.info("Performance Summary")
            self.logger.info("="*50)

            # Ensure we have the correct number of frames
            video_list = [results[i] for i in range(save_results)]
            video = np.concatenate(video_list, axis=0)

            video=video[:input_video_original.shape[2]]
            
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
    parser.add_argument("--flow_model", type=str, default=None, help="Optical flow model to use (from calflow). If None, flow is not calculated.")
    parser.add_argument("--x265_params", type=str, default='{"stage": "encode"}', help="x265 parameters as a JSON string. e.g., '{\"stage\": \"lookahead\"}'")
    parser.add_argument("--occlusion_method", type=str, default="quantile", choices=["exact","quantile", "morphological", "connected_components"], help="Method to generate occlusion mask.")
    parser.add_argument("--top_k_percentage", type=float, default=0.1, help="Top percentage of occlusion values to consider as masked.")
    parser.add_argument("--use_cached_text_embedding", action="store_true", help="If set, load pre-computed text embeddings from 'cached_text_embedding.pt' instead of initializing the text encoder.")
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    # Updated root logger to match target format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow_calculator = None

    if args.video_path is not None:
        ALIGNMENT = 32 
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if new_height != args.height or new_width != args.width:
            logging.warning(f"Adjusting resolution from {args.height}x{args.width} to {new_height}x{new_width}.")
        resize_hw = (new_height, new_width)
        args.height, args.width = new_height, new_width
        input_video_original, original_fps = load_mp4_as_tensor(args.video_path, resize_hw=resize_hw, device=device)

        args.fps=original_fps

        input_video_original = input_video_original.unsqueeze(0)
        logging.info(f"Input video tensor shape: {input_video_original.shape}")
        t = input_video_original.shape[2]
        input_video_original = input_video_original.to(dtype=torch.bfloat16)

        if args.flow_model!=None:
            logging.info(f"Preparing for optical flow calculation with model: {args.flow_model}")
            # flow_calculator = OpticalFlowCalculator(args.flow_model, device)
            x265_params = json.loads(args.x265_params)
            flow_calculator = OpticalFlowCalculator(
                flow_model_type=args.flow_model, 
                device=device, 
                x265_params=x265_params,
                occlusion_method=args.occlusion_method,
                top_k_percentage=args.top_k_percentage,
                # morph_kernel_size=args.morph_kernel_size,
                # conn_comp_threshold_quantile=args.conn_comp_thresh
            )
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
    if ((t-5)%chunk_size!=0): num_chunks+=1

    pipeline_manager = SingleGPUInferencePipeline(config, device, use_cached_text_embedding=args.use_cached_text_embedding)
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