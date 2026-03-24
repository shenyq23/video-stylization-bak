import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release"
)

import sys
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import logging
import cv2
import json

# --- Imports for models and data handling ---
# from causvid.models.wan.wan_wrapper import WanVAEWrapper
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange

# --- Add necessary paths to find project-specific modules ---
sys.path.append(os.path.join(os.path.dirname(__file__), "../StreamDiffusionV2"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../deps/gmflow"))

from test_3d_vae.test_vae import WanVAEWrapper

# --- Imports for optical flow ---
from utils.optical_wrapper import GMFlowWrapper, RAFTFlowWrapper, OcclusionComputation, X265MVWrapper
from gmflow.geometry import flow_warp as universal_flow_warp
from utils.vae_utils.mask_utils import build_gather_block_masks, dilate_mask, downsample_mask

from debugUtil import enable_custom_repr
enable_custom_repr()

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

def load_mp4_as_tensor(video_path: str, max_frames: int = None, resize_hw: tuple[int, int] = None, normalize: bool = True) -> tuple[torch.Tensor, int]:
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

    return video, int(original_fps)

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

    # 对 channel 维取 mean
    latent_mean = latent.mean(dim=0)

    min_val, max_val = latent_mean.min(), latent_mean.max()
    if max_val > min_val:
        latent_norm = (latent_mean - min_val) / (max_val - min_val)
    else:
        latent_norm = torch.zeros_like(latent_mean)

    img_np = (latent_norm.float().cpu().numpy() * 255).astype(np.uint8)
    # 把单通道灰度图，转换成 3 通道的 BGR 彩色图（三个通道的值完全一样）。
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)


# 在黑底画布上, 把光流张量画成“箭头图”
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
    # return cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)


def visualize_flow_with_source_overlay(source_image: np.ndarray, target_image: np.ndarray, flow_viz: np.ndarray, alpha: float = 0.4) -> np.ndarray:
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

        self.model = FlowModel(str(self.device), native_x265=True)
        
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
                # 为什么要 ×2？
                # 因为后面要做形态学操作，会删掉一部分像素, 所以前面先多选一点，留冗余
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
                    if covered_area >= target_area: 
                        break
                    region_mask_np = (labels_im == region['id'])
                    final_mask_np[region_mask_np] = True
                    covered_area += region['area']
                
                binary_mask = torch.from_numpy(final_mask_np).to(self.device)
            
            elif self.occlusion_method == 'gather_block':
                block_size = (6, 6)
                # stride = 4
                stride = 6

                H, W = single_occ_map.shape
                block_h, block_w = block_size

                # 1️⃣ 计算 block scores（sum）
                block_scores = (
                    F.avg_pool2d(
                        single_occ_map[None, None],  # [1,1,H,W]
                        block_size,
                        stride
                    )[0, 0] * (block_h * block_w)
                )

                H_out, W_out = block_scores.shape
                flat_scores = block_scores.view(-1)

                block_cnt = max(1, int(flat_scores.numel() * top_k_percentage))

                # 2️⃣ 找 top-k block 索引
                topk_idx = torch.topk(flat_scores, k=block_cnt, largest=True, sorted=False).indices

                # 3️⃣ block-level mask
                block_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
                block_mask[topk_idx] = True
                block_mask = block_mask.view(H_out, W_out)

                # 4️⃣ 映射回 pixel-level mask
                binary_mask = torch.zeros((H, W), dtype=torch.bool, device=single_occ_map.device)

                for i in range(H_out):
                    for j in range(W_out):
                        if block_mask[i, j]:
                            y0 = i * stride
                            x0 = j * stride
                            y1 = min(y0 + block_h, H)
                            x1 = min(x0 + block_w, W)
                            binary_mask[y0:y1, x0:x1] = True

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
        
        # print(f"ref frame: {ref_frame.mean():.4f}, current frame: {current_frame.mean():.4f}, fwd flow: {fwd_flow.mean():.4f}, bwd flow: {bwd_flow.mean():.4f}")

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

    @torch.no_grad()
    def _generate_visuals_loop(
        self,
        input_video: torch.Tensor,
        num_chunks: int,
        chunk_size: int,
        output_folder: str,
        top_k_percentage: float,
        vector_stride: int,
        occlusion_method: str,
        output_fps: int,
    ):
        self.logger.info("Visualization loop started.")
        start_frame_offset = 5

        viz_folder = os.path.join(output_folder, f"latent_warp_visualizations_{occlusion_method}_{top_k_percentage}")
        os.makedirs(viz_folder, exist_ok=True)

        self.logger.info(f"Priming VAE encoder with first {start_frame_offset} frames...")
        prime_frames = input_video[:, :, 0:start_frame_offset].to(self.device, dtype=torch.bfloat16)
        latents = self.vae.stream_encode(prime_frames)
        latents = latents.transpose(2, 1).contiguous()
        _ = self.vae.stream_decode_to_pixel(latents)
        self.logger.info("VAE encoder primed successfully.")

        decoded_video_chunks = []
        target_video_chunks = []
        decoded_frame_start = None
        decoded_frame_end = None

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

            bwd_flow, bwd_occ = flow_data
            
            chunk_n_frames = input_video[:, :, chunk_n_start_idx:chunk_n_end_idx]
            chunk_n1_frames = input_video[:, :, chunk_n1_start_idx:chunk_n1_end_idx]

            print(f"chunk_n_frames mean: {chunk_n_frames.mean():.4f}, chunk_n1_frames mean: {chunk_n1_frames.mean():.4f}")
            latent_n_raw = self.vae.stream_encode(chunk_n_frames.to(self.device, dtype=torch.bfloat16), is_full=True)
            latent_n = latent_n_raw.transpose(1, 2).contiguous()
            latent_n_repr = latent_n.squeeze(1)
            # latent_n_repr = latent_n
            
            _, _, latent_h, latent_w = latent_n_repr.shape

            # 改变mask分辨率
            downsampled_flow = F.interpolate(bwd_flow, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            # 改变mask数值
            downsampled_flow *= (float(latent_h) / bwd_flow.shape[2])
            print(f"latent_n mean: {latent_n_repr.mean():.4f}, downsampled_flow mean: {downsampled_flow.mean():.4f}")
            warped_latent = universal_flow_warp(latent_n_repr.float(), downsampled_flow.float())
            print(f"warped_latent mean: {warped_latent.mean():.4f}")

            # 改变occ分辨率
            downsampled_occ = F.interpolate(bwd_occ, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            
            # refine mask
            latent_binary_mask = self.flow_calculator.compute_binary_occlusion_mask(bwd_occ, top_k_percentage=top_k_percentage)[0, 0]
            latent_binary_mask_for_fused = self.flow_calculator.compute_binary_occlusion_mask(downsampled_occ, top_k_percentage=top_k_percentage)


            # Encoder masks
            mask_enc = dilate_mask(latent_binary_mask, 2)
            masks_enc = downsample_mask(mask_enc, min_res=(2, 2), dilation=2)
            # Decoder masks
            mask_dec = dilate_mask(latent_binary_mask, 2)
            masks_dec = downsample_mask(mask_dec, min_res=(2, 2), dilation=2)

            latent_n1_raw = self.vae.stream_encode(chunk_n1_frames.to(self.device, dtype=torch.bfloat16), mask=masks_enc, flow=bwd_flow, is_full=False)
            latent_n1 = latent_n1_raw.transpose(1, 2).contiguous()
            decoded_fused_video_chunk = self.vae.stream_decode_to_pixel(latent_n1, mask=masks_dec, flow=bwd_flow, is_full=False)
            


            latent_n1_gt = self.vae.stream_encode(chunk_n1_frames.to(self.device, dtype=torch.bfloat16), mask=None, flow=None, is_full=True)
            latent_n1_gt = latent_n1_gt.transpose(1, 2).contiguous()
            latent_n1_gt = latent_n1_gt.squeeze(1)

            # torch.where(condition, x, y)
            # 如果 condition[i] == True  → 取 x[i]
            # 如果 condition[i] == False → 取 y[i]
            latent_gt = torch.where(latent_binary_mask_for_fused, latent_n1_gt, warped_latent).to(dtype=latent_n.dtype)
            latent_gt_for_decode = latent_gt.unsqueeze(1)
            print(f"latent_gt_for_decode mean: {latent_gt_for_decode.mean():.4f}")
            decoded_fused_video_chunk_gt = self.vae.stream_decode_to_pixel(latent_gt_for_decode, None, None, is_full=True)


            
            decoded_chunk_btchw = decoded_fused_video_chunk.permute(0, 2, 1, 3, 4).contiguous()
            decoded_video_chunks.append(decoded_chunk_btchw.cpu())
            target_video_chunks.append(chunk_n1_frames)
            if decoded_frame_start is None:
                decoded_frame_start = chunk_n1_start_idx
            decoded_frame_end = chunk_n1_end_idx - 1

            # zeros_tensor_latent = torch.zeros_like(warped_latent)
            # occluded_warped_latent = torch.where(latent_binary_mask, zeros_tensor_latent, warped_latent)

            loss = torch.mean((latent_gt_for_decode - latent_n1)**2).sqrt()
            self.logger.info(f"Chunk {chunk_id}, top_k={top_k_percentage}, Latent RMSE: {loss.item():.4f}")

            # # --- 1. Latent Space Visualization ---
            # viz_latent_n_mean = visualize_latent_to_image(latent_n_repr)
            # viz_latent_n1_mean = visualize_latent_to_image(latent_n1_repr)
            # # viz_warped_latent_mean = visualize_latent_to_image(occluded_warped_latent)
            # viz_fused_latent_mean = visualize_latent_to_image(fused_latent_repr)
            # latent_stride = max(5, vector_stride // 4)
            # viz_latent_flow = visualize_flow_to_rgb(downsampled_flow, vector_stride=latent_stride)
            # viz_compo=visualize_flow_with_source_overlay(viz_latent_n_mean, viz_latent_n1_mean, viz_latent_flow, alpha=0.4)
            
            # # viz_overlayed_latent_flow = overlay_flow_on_image(viz_latent_n1_mean, viz_latent_flow)
            
            # # mean_latent_row = np.concatenate([viz_latent_n_mean, viz_latent_n1_mean, viz_warped_latent_mean, viz_fused_latent_mean, viz_compo], axis=1)
            # mean_latent_row = np.concatenate([viz_latent_n_mean, viz_latent_n1_mean, viz_fused_latent_mean, viz_compo], axis=1)


            # # --- 2. Pixel Space Visualization ---
            # pixel_rows = []
            num_frames_in_chunk = chunk_n_frames.shape[2]
            # pixel_binary_mask = self.flow_calculator.compute_binary_occlusion_mask(bwd_occ, top_k_percentage=top_k_percentage)
            # ratio = pixel_binary_mask.float().mean().item()
            # self.logger.info(f"pixel_binary_mask 中为 1 的比例: {ratio*100:.2f}%")
            
            warped_frames = []
            decoded_fused_frames = []
            target_frames = []
            
            for j in range(num_frames_in_chunk):
                frame_n = chunk_n_frames[:, :, j].to(self.device)
                frame_n1 = chunk_n1_frames[:, :, j].to(self.device)

                # warped_frame = universal_flow_warp(frame_n.float(), bwd_flow.float())
                # green_color = torch.tensor([-1.0, 1.0, -1.0], device=warped_frame.device, dtype=warped_frame.dtype).view(1, 3, 1, 1)
                # warped_frame = torch.where(pixel_binary_mask, green_color, warped_frame)
                
                # viz_frame_n = tensor_to_np_img(frame_n)
                # viz_frame_n1 = tensor_to_np_img(frame_n1)
                # # viz_warped_frame = tensor_to_np_img(occluded_warped_frame)
                # viz_warped_frame = tensor_to_np_img(warped_frame)
                # # print(i,j,torch.mean(frame_n),torch.mean(frame_n1),torch.mean(bwd_flow),torch.mean(warped_frame))
                decoded_fused_frame = decoded_fused_video_chunk[:, j]
                decoded_fused_frame_gt = decoded_fused_video_chunk_gt[:, j]
                # viz_decoded_fused_frame = tensor_to_np_img(decoded_fused_frame)
                # viz_frame_flow = visualize_flow_to_rgb(bwd_flow, vector_stride=vector_stride)

                # # viz_overlayed_pixel_flow = overlay_flow_on_image(viz_frame_n1, viz_frame_flow)
                # viz_frame_compo=visualize_flow_with_source_overlay(viz_frame_n, viz_frame_n1, viz_frame_flow, alpha=0.4)
                
                # pixel_row = np.concatenate([viz_frame_n, viz_frame_n1, viz_warped_frame, viz_decoded_fused_frame, viz_frame_compo], axis=1)
                # pixel_rows.append(pixel_row)

                # psnr_pixel_vs_target = calculate_psnr(warped_frame, frame_n1)
                psnr_latent_vs_target = calculate_psnr(decoded_fused_frame, frame_n1)
                psnr_latent_gt_vs_target = calculate_psnr(decoded_fused_frame_gt, frame_n1)
                if j == 3:
                    self.logger.info(f"  Chunk {chunk_id} Frame {j} Quality Metrics:")
                # self.logger.info(f"pixel  | vs Target (预测目标): {psnr_pixel_vs_target:.2f} dB")
                    self.logger.info(f"latent | vs Target (预测目标): {psnr_latent_vs_target:.2f} dB")
                    self.logger.info(f"latent_gt | vs Target (预测目标): {psnr_latent_gt_vs_target:.2f} dB")

            # calculate Chunk-level PSNR
                # warped_frames.append(warped_frame)
                # decoded_fused_frames.append(decoded_fused_frame)
                # target_frames.append(frame_n1)
            # chunk_warped = torch.stack(warped_frames, dim=2)
            # chunk_decoded = torch.stack(decoded_fused_frames, dim=2)
            # chunk_target = torch.stack(target_frames, dim=2)
            # psnr_pixel_vs_target = calculate_psnr(chunk_warped, chunk_target)
            # psnr_latent_vs_target = calculate_psnr(chunk_decoded, chunk_target)
            # self.logger.info(f"  Chunk {chunk_id} Quality Metrics (chunk-level):")
            # # self.logger.info(f"pixel  | vs Target (预测目标): {psnr_pixel_vs_target:.2f} dB")
            # self.logger.info(f"latent | vs Target (预测目标): {psnr_latent_vs_target:.2f} dB")


            # # --- 3. Combine and Save ---
            # target_width = pixel_rows[0].shape[1]
            # latent_h, latent_w, _ = mean_latent_row.shape
            # new_latent_h = int(latent_h * (target_width / latent_w))
            
            # resized_latent_block = cv2.resize(mean_latent_row, (target_width, new_latent_h), interpolation=cv2.INTER_NEAREST)

            # final_image = np.concatenate([resized_latent_block] + pixel_rows, axis=0)
            
            # save_path = os.path.join(viz_folder, f"chunk_{chunk_id:04d}_{occlusion_method}_{top_k_percentage}.png")
            # cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            # self.logger.info(f"Saved visualization for chunk {chunk_id} to {save_path}")


        decoded_video = torch.cat(decoded_video_chunks, dim=2)
        target_video = torch.cat(target_video_chunks, dim=2)
        psnr_full = calculate_psnr(decoded_video, target_video)
        self.logger.info(
            f"Full decoded video PSNR vs input: {psnr_full:.2f} dB "
            f"(frames {decoded_frame_start}-{decoded_frame_end}, total {decoded_video.shape[2]})"
        )

        save_path = os.path.join(
            output_folder,
            f"decoded_video_{occlusion_method}_{top_k_percentage}.mp4",
        )
        decoded_video_for_save = (
            (decoded_video[0] * 0.5 + 0.5)
            .clamp(0, 1)
            .permute(1, 2, 3, 0)
            .mul(255)
            .byte()
            .cpu()
        )
        torchvision.io.write_video(save_path, decoded_video_for_save, fps=output_fps)
        self.logger.info(f"Saved decoded video to {save_path}")

        self.logger.info("Visualization loop finished.")

    def run_visualization(
        self,
        input_video: torch.Tensor,
        output_folder: str,
        chunk_size: int,
        top_k_percentage: float,
        vector_stride: int,
        occlusion_method: str,
        output_fps: float,
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
            occlusion_method,
            output_fps,
        )

        self.logger.info("Visualization pipeline completed successfully.")

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Latent and Pixel Space Warping Visualization")
    parser.add_argument("--video_path", type=str, default="data/bird.mp4", help="Path to the original input video file.")
    parser.add_argument("--output_folder", type=str, default="data/output", help="Folder to save the visualization images.")
    parser.add_argument("--flow_model", type=str, default="x265", choices=["gmflow", "raft", "x265", "none"], help="Optical flow model.")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type for WanVAEWrapper (e.g., T2V-1.3B)")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--chunk_size", type=int, default=4, help="Frames per chunk.")
    parser.add_argument("--x265_params", type=str, default='{"stage":"encode", "quiet":true}', help="x265 parameters as a JSON string. e.g., '{\"stage\": \"lookahead\"}'")
    parser.add_argument("--occlusion_method", type=str, default="quantile", choices=["quantile", "morphological", "connected_components", "gather_block"], help="Method to generate occlusion mask.")
    parser.add_argument("--top_k_percentage", type=float, default=0.1, help="Top percentage of occlusion values to consider as masked.")
    parser.add_argument("--morph_kernel_size", type=int, default=7, help="Kernel size for morphological opening operation.")
    parser.add_argument("--conn_comp_thresh", type=float, default=0.75, help="Initial quantile threshold for connected components analysis.")
    parser.add_argument("--vector_stride", type=int, default=20, help="Stride for drawing flow vectors in the arrow visualization.")
    parser.add_argument("--log_file", type=str, default="", help="Path to save logs. Default: <output_folder>/run.log")
    def _kernel_backend(v: str) -> str:
        v = (v or "").strip().lower()
        if v in {"pytorch", "torch"}:
            return "pytorch"
        if v in {"cuda", "ext"}:
            return "cuda"
        raise argparse.ArgumentTypeError("Expected 'PyTorch' or 'CUDA'.")

    parser.add_argument(
        "--sige_kernels",
        type=_kernel_backend,
        default="cuda",
        help="SIGE gather/scatter kernel backend: PyTorch (default) or CUDA.",
    )
    
    args = parser.parse_args()

    from sige3d.torch_kernels.backend import set_kernel_backend
    set_kernel_backend(args.sige_kernels)

    torch.set_grad_enabled(False)
    os.makedirs(args.output_folder, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_folder, f"{args.occlusion_method}_{args.top_k_percentage}_run.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )

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
    input_video_tensor, original_fps = load_mp4_as_tensor(args.video_path, resize_hw=resize_hw, max_frames=args.max_frames)
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
            occlusion_method=args.occlusion_method,
            output_fps=original_fps,
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
