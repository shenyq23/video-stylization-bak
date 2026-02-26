import sys
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import logging
import cv2
import json

# --- Add necessary paths to find project-specific modules ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../StreamDiffusionV2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../deps/gmflow"))


# --- Imports for models and data handling ---
from causvid.models.wan.wan_wrapper import WanVAEWrapper
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# --- Imports for optical flow ---
from utils.optical_wrapper import GMFlowWrapper, RAFTFlowWrapper, OcclusionComputation, X265MVWrapper
from gmflow.geometry import flow_warp as universal_flow_warp

# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算两张图像之间的峰值信噪比 (PSNR)。
    假设输入张量的范围是 [-1, 1]。
    """
    # 将图像范围从 [-1, 1] 转换为 [0, 1] 以进行标准 PSNR 计算
    img1 = (img1 * 0.5 + 0.5).clamp(0, 1)
    img2 = (img2 * 0.5 + 0.5).clamp(0, 1)
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
    """Loads a video file into a [C, T, H, W] tensor."""
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video, _, info = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    original_fps = info.get('video_fps', 30)

    if max_frames is not None:
        video = video[:max_frames]

    video = rearrange(video, "t c h w -> c t h w")

    if resize_hw is not None:
        resized_frames = [TF.resize(video[:, i], resize_hw, antialias=True) for i in range(video.shape[1])]
        video = torch.stack(resized_frames, dim=1)

    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video, original_fps

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

# --- MODIFICATION START: New helper function for overlaying flow ---
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
# --- MODIFICATION END ---

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

# ==============================================================================
# Core Classes
# ==============================================================================

class OpticalFlowCalculator:
    def __init__(self, 
                 flow_model_type: str, 
                 device: torch.device, 
                 x265_params: dict = None,
                 occlusion_method: str = 'quantile',
                 morph_kernel_size: int = 7,
                 conn_comp_threshold_quantile: float = 0.75
                ):
        self.device = device
        self.logger = logging.getLogger("OpticalFlowCalculator")
        self.x265_params = x265_params or {}
        self.flow_model_type = flow_model_type
        self.occlusion_method = occlusion_method
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

        self.model = FlowModel(str(self.device),native_x265=True)
        
        if self.flow_model_type.lower() == 'x265':
             self.logger.info("Using 'luminosity' occlusion for x265.")
             self.occlusion_computer = OcclusionComputation(use_luminosity=True)
        else:
             self.logger.info("Using 'geometry' occlusion for DL models.")
             self.occlusion_computer = OcclusionComputation(use_geometry=True)


    def compute_binary_occlusion_mask(self, raw_occ_map: torch.Tensor, top_k_percentage: float) -> torch.Tensor:
        B, _, H, W = raw_occ_map.shape
        final_masks = []

        for i in range(B):
            single_occ_map = raw_occ_map[i, 0]

            if self.occlusion_method == 'quantile':
                threshold = torch.quantile(single_occ_map, 1.0 - top_k_percentage)
                binary_mask = (single_occ_map >= threshold)
            elif self.occlusion_method == 'morphological':
                initial_quantile = max(0.5, 1.0 - top_k_percentage * 2)
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
                target_area = H * W * top_k_percentage
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

class LatentWarpVisualizer:
    def __init__(self, vae: WanVAEWrapper, flow_calculator: OpticalFlowCalculator, device: torch.device):
        self.vae = vae
        self.device = device
        self.logger = logging.getLogger("LatentWarpVisualizer")
        self.flow_calculator = flow_calculator
        self.prev_latent=None

    def _optimize_latent_flow(
        self,
        latent_n_repr: torch.Tensor,
        latent_n1_repr: torch.Tensor,
        target_pixel_frames: torch.Tensor,
        latent_binary_mask: torch.Tensor,
        initial_flow: torch.Tensor,
        optim_steps: int=100,
        optim_lr: float=0.01,
    ) -> torch.Tensor:
        """
        通过梯度下降优化，计算一个最优的潜在空间光流。
        """
        self.logger.info(f"Starting latent flow optimization for {optim_steps} steps with lr={optim_lr}...")

        latent_n_repr = latent_n_repr.detach()
        latent_n1_repr = latent_n1_repr.detach()
        target_pixel_frames = target_pixel_frames.detach()
        latent_binary_mask = latent_binary_mask.detach()
        initial_flow = initial_flow.detach()

        # 1. 初始化一个可学习的 latent_flow，以 initial_flow 为起点
        latent_flow = initial_flow.clone().requires_grad_(True)
        
        # 2. 设置优化器
        optimizer = torch.optim.Adam([latent_flow], lr=optim_lr)
        
        # 确保 VAE 模型参数不参与梯度计算
        # self.vae.eval()

        # 3. 优化循环
        # 使用 torch.set_grad_enabled(True) 确保在这个函数内可以计算梯度
        with torch.set_grad_enabled(True):
            for step in range(optim_steps):
                optimizer.zero_grad()
                
                # a. 使用当前 latent_flow 进行 warp
                warped_latent = universal_flow_warp(latent_n_repr, latent_flow)
                
                # b. 融合 latent
                fused_latent = torch.where(latent_binary_mask, latent_n1_repr, warped_latent).to(self.device, dtype=torch.bfloat16)
                
                # c. 解码到像素空间
                # VAE解码需要 (B, T, C, H, W) 格式
                decoded_frames = self.vae.stream_decode_to_pixel(fused_latent.unsqueeze(1))

                decoded_frames=decoded_frames.transpose(2,1).contiguous()

                # print(decoded_frames.shape,target_pixel_frames.shape)
                
                # d. 计算损失 (与真实目标帧的 MSE)
                loss = F.mse_loss(decoded_frames.float(), target_pixel_frames.float())
                
                # e. 反向传播和优化
                loss.backward()
                optimizer.step()

                latent_flow.detach_()
                latent_n_repr.detach_()
                
                self.logger.info(f"  Optim Step [{step+1}/{optim_steps}], Loss: {loss.item():.6f}")

        self.logger.info(f"Optimization finished. Final Loss: {loss.item():.6f}")
        
        # 返回优化后的、分离了计算图的 latent_flow
        return latent_flow.detach().float()

    def _generate_visuals_loop(
        self,
        input_video: torch.Tensor,
        num_chunks: int,
        chunk_size: int,
        output_folder: str, 
        top_k_percentage: float,
        vector_stride: int,
    ):
        self.logger.info("Visualization loop started.")
        start_frame_offset = 5

        viz_folder = os.path.join(output_folder, f"latent_warp_visualizations_{top_k_percentage}")
        os.makedirs(viz_folder, exist_ok=True)

        self.logger.info(f"Priming VAE encoder with first {start_frame_offset} frames...")
        prime_frames = input_video[:, :, 0:start_frame_offset].to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = self.vae.stream_encode(prime_frames)
            latents = latents.transpose(2, 1).contiguous()
            _ = self.vae.stream_decode_to_pixel(latents)
        self.logger.info("VAE encoder primed successfully.")

        self.prev_latent=None

        for i in range(num_chunks - 1):
            chunk_id = i + 1
            chunk_n_start_idx = start_frame_offset + i * chunk_size
            chunk_n_end_idx = chunk_n_start_idx + chunk_size
            chunk_n1_start_idx = chunk_n_end_idx
            chunk_n1_end_idx = chunk_n1_start_idx + chunk_size

            if chunk_n1_end_idx > input_video.shape[2]:
                self.logger.warning(f"Not enough frames for chunk {chunk_id+1}. Stopping.")
                break

            ref_frame_idx = chunk_n_end_idx - 1
            current_frame_idx = chunk_n1_end_idx - 1
            self.logger.info(f"Chunk {chunk_id}: calculating flow Frame {ref_frame_idx} -> {current_frame_idx}")

            ref_frame = input_video[:, :, ref_frame_idx:ref_frame_idx+1].squeeze(2).to(self.device)
            current_frame = input_video[:, :, current_frame_idx:current_frame_idx+1].squeeze(2).to(self.device)

            flow_data = self.flow_calculator.calculate_flow(ref_frame, current_frame)
            if not flow_data: continue
            
            bwd_flow, bwd_occ = flow_data
            
            chunk_n_frames = input_video[:, :, chunk_n_start_idx:chunk_n_end_idx]
            chunk_n1_frames = input_video[:, :, chunk_n1_start_idx:chunk_n1_end_idx]

            with torch.no_grad():
                if self.prev_latent==None: 
                    latent_n_raw = self.vae.stream_encode(chunk_n_frames.to(self.device, dtype=torch.bfloat16))
                    latent_n = latent_n_raw.transpose(1, 2).contiguous()
                else: 
                    # latent_n_raw = self.vae.stream_encode(chunk_n_frames.to(self.device, dtype=torch.bfloat16))
                    latent_n=self.prev_latent
                latent_n1_raw = self.vae.stream_encode(chunk_n1_frames.to(self.device, dtype=torch.bfloat16))
                latent_n1 = latent_n1_raw.transpose(1, 2).contiguous()

            latent_n_repr = latent_n.squeeze(1)
            latent_n1_repr = latent_n1.squeeze(1)
            _, _, latent_h, latent_w = latent_n_repr.shape


            optimize_flow=True

            downsampled_occ = F.interpolate(bwd_occ, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            latent_binary_mask = self.flow_calculator.compute_binary_occlusion_mask(downsampled_occ, top_k_percentage=top_k_percentage)

            downsampled_flow = F.interpolate(bwd_flow, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            downsampled_flow *= (float(latent_h) / bwd_flow.shape[2])

            if optimize_flow:
                # --- 新路径：通过优化计算 latent flow ---
                # 注意：将输入转为 float32 以保证梯度计算的稳定性
                optimized_latent_flow = self._optimize_latent_flow(
                    latent_n_repr=latent_n_repr.float().detach(),
                    latent_n1_repr=latent_n1_repr.float().detach(),
                    target_pixel_frames=chunk_n1_frames.to(self.device).detach(),
                    latent_binary_mask=latent_binary_mask.detach(),
                    initial_flow=downsampled_flow.float().detach()
                )
                downsampled_flow = optimized_latent_flow
            else:
                # --- 旧路径：通过下采样计算 latent flow ---
                self.logger.info("Using downsampled pixel flow for latent warping.")
                
                






            warped_latent = universal_flow_warp(latent_n_repr.float(), downsampled_flow.float())
            
            fused_latent_repr = torch.where(latent_binary_mask, latent_n1_repr, warped_latent).to(dtype=latent_n.dtype)

            with torch.no_grad():
                fused_latent_for_decode = fused_latent_repr.unsqueeze(1)
                decoded_fused_video_chunk = self.vae.stream_decode_to_pixel(fused_latent_for_decode)

            fill_tensor_latent = torch.zeros_like(warped_latent)
            occluded_warped_latent = torch.where(latent_binary_mask.bool(), fill_tensor_latent, warped_latent)

            loss = torch.mean((fused_latent_repr - latent_n1_repr)**2).sqrt()
            self.logger.info(f"Chunk {chunk_id}, top_k={top_k_percentage}, Latent RMSE: {loss.item():.4f}")

            # --- 1. Latent Space Visualization ---
            viz_latent_n_mean = visualize_latent_to_image(latent_n_repr)
            viz_latent_n1_mean = visualize_latent_to_image(latent_n1_repr)
            viz_warped_latent_mean = visualize_latent_to_image(occluded_warped_latent)
            viz_fused_latent_mean = visualize_latent_to_image(fused_latent_repr)
            latent_stride = max(5, vector_stride // 4)
            viz_latent_flow = visualize_flow_to_rgb(downsampled_flow, vector_stride=latent_stride)
            viz_compo=visualize_flow_with_source_overlay(viz_latent_n_mean, viz_latent_n1_mean, viz_latent_flow, alpha=0.4)
            
            # --- MODIFICATION: Create the new overlay column for latent space ---
            # viz_overlayed_latent_flow = overlay_flow_on_image(viz_latent_n1_mean, viz_latent_flow)
            
            mean_latent_row = np.concatenate([viz_latent_n_mean, viz_latent_n1_mean, viz_warped_latent_mean, viz_fused_latent_mean, viz_compo], axis=1)
            
            # --- 2. Pixel Space Visualization ---
            pixel_rows = []
            num_frames_in_chunk = chunk_n_frames.shape[2]
            pixel_binary_mask = self.flow_calculator.compute_binary_occlusion_mask(bwd_occ, top_k_percentage=top_k_percentage)
            
            for j in range(num_frames_in_chunk):
                frame_n = chunk_n_frames[:, :, j].to(self.device)
                frame_n1 = chunk_n1_frames[:, :, j].to(self.device)
                warped_frame = universal_flow_warp(frame_n.float(), bwd_flow.float())
                occluded_warped_frame = torch.where(
                    pixel_binary_mask.to(device=warped_frame.device), 
                    torch.zeros_like(warped_frame), 
                    warped_frame
                )

                viz_frame_n = tensor_to_np_img(frame_n)
                viz_frame_n1 = tensor_to_np_img(frame_n1)
                # viz_warped_frame = tensor_to_np_img(occluded_warped_frame)
                viz_warped_frame = tensor_to_np_img(warped_frame)
                # print(i,j,torch.mean(frame_n),torch.mean(frame_n1),torch.mean(bwd_flow),torch.mean(warped_frame))
                decoded_fused_frame = decoded_fused_video_chunk[:, j]
                viz_decoded_fused_frame = tensor_to_np_img(decoded_fused_frame)
                viz_frame_flow = visualize_flow_to_rgb(bwd_flow, vector_stride=vector_stride)

                # --- MODIFICATION: Create the new overlay column for pixel space ---
                # viz_overlayed_pixel_flow = overlay_flow_on_image(viz_frame_n1, viz_frame_flow)
                viz_frame_compo=visualize_flow_with_source_overlay(viz_frame_n, viz_frame_n1, viz_frame_flow, alpha=0.4)
                
                pixel_row = np.concatenate([viz_frame_n, viz_frame_n1, viz_warped_frame, viz_decoded_fused_frame, viz_frame_compo], axis=1)
                pixel_rows.append(pixel_row)

                psnr_pixel_vs_target = calculate_psnr(warped_frame, frame_n1)
                psnr_latent_vs_target = calculate_psnr(decoded_fused_frame, frame_n1)
                if j == 3: 
                    self.logger.info(f"  Chunk {chunk_id} Frame {j} Quality Metrics:")
                    self.logger.info(f"     | vs Target (预测目标): {psnr_pixel_vs_target:.2f} dB")
                    self.logger.info(f"     | vs Target (预测目标): {psnr_latent_vs_target:.2f} dB")

            # --- 3. Combine and Save ---
            target_width = pixel_rows[0].shape[1]
            latent_h, latent_w, _ = mean_latent_row.shape
            new_latent_h = int(latent_h * (target_width / latent_w))
            
            resized_latent_block = cv2.resize(mean_latent_row, (target_width, new_latent_h), interpolation=cv2.INTER_NEAREST)

            final_image = np.concatenate([resized_latent_block] + pixel_rows, axis=0)
            
            save_path = os.path.join(viz_folder, f"comparison_chunk_{chunk_id:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Saved visualization for chunk {chunk_id} to {save_path}")

            self.prev_latent=fused_latent_repr.unsqueeze(1).detach().clone()

        self.logger.info("Visualization loop finished.")

    def run_visualization(
        self,
        input_video: torch.Tensor,
        output_folder: str,
        chunk_size: int,
        top_k_percentage: float,
        vector_stride: int,
    ):
        self.logger.info("Starting Latent Warping Visualization Pipeline...")
        os.makedirs(output_folder, exist_ok=True)

        num_chunks = (input_video.shape[2] - 5) // chunk_size
        if num_chunks <= 1:
            self.logger.error(f"Not enough frames for visualization. Need at least {5 + 2*chunk_size} frames.")
            return
        
        # Simplified direct execution
        self._generate_visuals_loop(
            input_video, 
            num_chunks, 
            chunk_size, 
            output_folder, 
            top_k_percentage,
            vector_stride,
        )

        self.logger.info("Visualization pipeline completed successfully.")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Latent and Pixel Space Warping Visualization")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the original input video file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the visualization images.")
    parser.add_argument("--flow_model", type=str, default="gmflow", choices=["gmflow", "raft", "x265", "none"], help="Optical flow model.")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type for WanVAEWrapper (e.g., T2V-1.3B)")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--chunk_size", type=int, default=4, help="Frames per chunk.")
    parser.add_argument("--x265_params", type=str, default='{}', help="x265 parameters as a JSON string. e.g., '{\"stage\": \"lookahead\"}'")
    parser.add_argument("--occlusion_method", type=str, default="quantile", choices=["quantile", "morphological", "connected_components"], help="Method to generate occlusion mask.")
    parser.add_argument("--top_k_percentage", type=float, default=0.1, help="Top percentage of occlusion values to consider as masked.")
    parser.add_argument("--morph_kernel_size", type=int, default=7, help="Kernel size for morphological opening operation.")
    parser.add_argument("--conn_comp_thresh", type=float, default=0.75, help="Initial quantile threshold for connected components analysis.")
    parser.add_argument("--vector_stride", type=int, default=30, help="Stride for drawing flow vectors in the arrow visualization.")
    args = parser.parse_args()

    # torch.set_grad_enabled(False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        x265_params = json.loads(args.x265_params)
        logging.info(f"Parsed x265 parameters: {x265_params}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format for --x265_params: {args.x265_params}")
        sys.exit(1)

    logging.info(f"Initializing WanVAEWrapper with model_type: {args.model_type}")
    try:
        vae = WanVAEWrapper(model_type=args.model_type)
        vae.to(device, dtype=torch.bfloat16)
        logging.info("WanVAEWrapper initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize WanVAEWrapper: {e}", exc_info=True)
        sys.exit(1)

    ALIGNMENT = 32
    resize_hw = ((args.height // ALIGNMENT) * ALIGNMENT, (args.width // ALIGNMENT) * ALIGNMENT)
    input_video_tensor, original_fps = load_mp4_as_tensor(args.video_path, resize_hw=resize_hw)
    input_video_tensor = input_video_tensor.unsqueeze(0)
    logging.info(f"Loaded video tensor with shape: {input_video_tensor.shape}")

    # x265_params['frame_rate'] = original_fps

    flow_calculator = OpticalFlowCalculator(
        flow_model_type=args.flow_model, 
        device=device, 
        x265_params=x265_params,
        occlusion_method=args.occlusion_method,
        morph_kernel_size=args.morph_kernel_size,
        conn_comp_threshold_quantile=args.conn_comp_thresh
    )

    visualizer = LatentWarpVisualizer(vae, flow_calculator, device)

    try:
        visualizer.run_visualization(
            input_video=input_video_tensor,
            output_folder=args.output_folder,
            chunk_size=args.chunk_size,
            top_k_percentage=args.top_k_percentage,
            vector_stride=args.vector_stride,
        )
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}", exc_info=True)
        raise

    logging.info("="*50)
    logging.info("Script finished.")
    logging.info(f"Check the '{os.path.join(args.output_folder, 'latent_warp_visualizations_*')}' directory for output images.")
    logging.info("="*50)

if __name__ == "__main__":
    main()