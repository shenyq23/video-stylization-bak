import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import sys
sys.path.insert(0, "..")
sys.path.append(os.path.join(os.path.dirname(__file__), "../deps/gmflow"))
import argparse
from gmflow.geometry import flow_warp
import torchvision.transforms.functional as TF

# --- 修改点 1: 导入 X265MVWrapper ---
# 我们不再直接使用 X265NativeWrapper，而是通过它的父类来调用
# 这样我们就可以通过参数选择使用哪个后端
try:
    # 假设 test.py 和 optical_wrapper.py 在同一个目录下 (utils)
    from utils.optical_wrapper import X265MVWrapper, GMFlowWrapper
except ImportError as e:
    print(f"错误: 无法导入 X265MVWrapper。请确保 optical_wrapper.py 文件与此脚本在同一目录中。")
    print(f"原始错误: {e}")
    exit()

# --- flow_warp 及其依赖 (保持不变) ---
# def coords_grid(b, h, w, device='cpu'):
#     coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
#     coords = torch.stack(coords[::-1], dim=0).float()
#     return coords.unsqueeze(0).repeat(b, 1, 1, 1)

# def bilinear_sample(feature, grid, padding_mode='zeros', return_mask=False):
#     h, w = feature.shape[-2:]
#     grid_normalized = grid.clone()
#     grid_normalized[:, 0, :, :] = 2.0 * grid_normalized[:, 0, :, :] / max(w - 1, 1) - 1.0
#     grid_normalized[:, 1, :, :] = 2.0 * grid_normalized[:, 1, :, :] / max(h - 1, 1) - 1.0
#     grid_normalized = grid_normalized.permute(0, 2, 3, 1)
#     output = F.grid_sample(feature, grid_normalized, padding_mode=padding_mode, align_corners=True)
#     if return_mask:
#         mask = (grid_normalized[..., 0] >= -1) & (grid_normalized[..., 0] <= 1) & \
#                (grid_normalized[..., 1] >= -1) & (grid_normalized[..., 1] <= 1)
#         return output, mask.unsqueeze(1)
#     return output

# def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
#     b, c, h, w = feature.size()
#     assert flow.size(1) == 2, f"Flow tensor must have 2 channels, but got {flow.size(1)}"
#     grid = coords_grid(b, h, w, device=flow.device) + flow
#     return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)
# --- 结束 flow_warp 部分 ---

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
    res=cv2.cvtColor(flow_canvas, cv2.COLOR_BGR2RGB)
    res=torch.from_numpy(res).permute(2,0,1).unsqueeze(0).float()/255.0
    return res

# --- 光流可视化函数 (保持不变) ---
def flow_to_color(flow: torch.Tensor, max_flow: float = None):
    n, _, h, w = flow.shape
    u, v = flow[:, 0, :, :], flow[:, 1, :, :]
    mag = torch.sqrt(u**2 + v**2)
    angle = torch.atan2(v, u)
    hsv_h = (angle / (2 * np.pi)) + 0.5
    hsv_s = torch.ones_like(hsv_h)
    if max_flow is not None:
        hsv_v = torch.clamp(mag / max_flow, 0, 1)
    else:
        mag_min, mag_max = mag.min(), mag.max()
        hsv_v = (mag - mag_min) / (mag_max - mag_min) if mag_max > mag_min else torch.zeros_like(mag)
    hsv_h_ = hsv_h * 6
    i = torch.floor(hsv_h_).long()
    f = hsv_h_ - i
    p = hsv_v * (1 - hsv_s)
    q = hsv_v * (1 - f * hsv_s)
    t = hsv_v * (1 - (1 - f) * hsv_s)
    r, g, b = torch.zeros_like(hsv_h), torch.zeros_like(hsv_h), torch.zeros_like(hsv_h)
    mask0, mask1, mask2, mask3, mask4, mask5 = (i == 0) | (i == 6), i == 1, i == 2, i == 3, i == 4, i == 5
    r[mask0], g[mask0], b[mask0] = hsv_v[mask0], t[mask0], p[mask0]
    r[mask1], g[mask1], b[mask1] = q[mask1], hsv_v[mask1], p[mask1]
    r[mask2], g[mask2], b[mask2] = p[mask2], hsv_v[mask2], t[mask2]
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], hsv_v[mask3]
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], hsv_v[mask4]
    r[mask5], g[mask5], b[mask5] = hsv_v[mask5], p[mask5], q[mask5]
    return torch.stack([r, g, b], dim=1)


def create_shifted_frames(image, shift_pixels: int):
    # if not os.path.exists(image_path):
    #     raise FileNotFoundError(f"图片文件未找到: {image_path}")
    # source_bgr = cv2.imread(image_path)
    # if source_bgr is None:
    #     raise IOError(f"无法读取图片: {image_path}")
    source_bgr = image
    h, w, _ = source_bgr.shape
    translation_matrix = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    target_bgr = cv2.warpAffine(source_bgr, translation_matrix, (w, h), borderValue=(0, 0, 0))
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    # 将图片从 0-255 (uint8) 转换为 -1 到 1 (float32) 的 Tensor
    source_tensor = (torch.from_numpy(source_rgb.astype(np.float32)).permute(2, 0, 1) / 127.5) - 1.0
    target_tensor = (torch.from_numpy(target_rgb.astype(np.float32)).permute(2, 0, 1) / 127.5) - 1.0
    return source_tensor.unsqueeze(0), target_tensor.unsqueeze(0)


def run_verification_and_save_images(source_tensor: torch.Tensor, target_tensor: torch.Tensor, output_dir: str = "warp_results_from_png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # [步骤 1] 保存传入的源帧和目标帧
    print(f"\n[步骤 1] 正在保存从PNG加载的源帧和目标帧...")
    source_tensor, target_tensor = source_tensor.to(device), target_tensor.to(device)
    save_image(source_tensor * 0.5 + 0.5, os.path.join(output_dir, "01_source_frame.png"))
    save_image(target_tensor * 0.5 + 0.5, os.path.join(output_dir, "02_target_frame.png"))
    print(" -> 源帧和目标帧已保存。")

    print(f"\n[步骤 2] 正在计算 Source -> Target 之间的光流...")
    try:
        # 使用 GMFlow 计算光流 (这部分逻辑保持不变)
        # flow_computer = GMFlowWrapper(device)
        # forward_flow, backward_flow = flow_computer.compute_flow_from_tensors(
        #     ref_frame_tensor=source_tensor,
        #     current_frame_tensor=target_tensor,
        # )
        params={}
        params["stage"]="encode"
        # params["frame_rate"]=25.0
        flow_computer=X265MVWrapper(device=device,native_x265=False)
        import time
        start_time=time.time()
        forward_flow, backward_flow = flow_computer.compute_flow_from_tensors(
            ref_frame_tensor=source_tensor,
            current_frame_tensor=target_tensor,
            **params
        )
        end_time=time.time()
        print(f"光流计算时间: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"计算光流时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    print(" -> 光流计算完成。")
    
    print("\n[步骤 3] 分析光流具体数值...")
    _, _, h, w = source_tensor.shape
    center_y, center_x = h // 2, w // 2
    
    # 检查中心区域的平均光流值，因为真实图像运动复杂
    patch_size = 10
    center_patch_fwd = forward_flow[0, :, center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size]
    center_patch_bwd = backward_flow[0, :, center_y-patch_size:center_y+patch_size, center_x-patch_size:center_x+patch_size]
    
    avg_fwd_flow = torch.mean(center_patch_fwd, dim=[1, 2]).cpu().numpy()
    avg_bwd_flow = torch.mean(center_patch_bwd, dim=[1, 2]).cpu().numpy()

    print(f"  - 图像尺寸: {w}x{h}")
    print(f"  - 中心区域平均 Forward Flow (u, v): ({avg_fwd_flow[0]:.2f}, {avg_fwd_flow[1]:.2f})")
    print(f"  - 中心区域平均 Backward Flow (u, v): ({avg_bwd_flow[0]:.2f}, {avg_bwd_flow[1]:.2f})")

    print("\n[步骤 4] 正在可视化光流场...")
    # 动态确定 max_flow 以获得更好的可视化效果
    max_flow_val = max(torch.max(torch.abs(forward_flow)).item(), torch.max(torch.abs(backward_flow)).item(), 1.0)
    # fwd_flow_vis = flow_to_color(forward_flow, max_flow=max_flow_val)
    # bwd_flow_vis = flow_to_color(backward_flow, max_flow=max_flow_val)
    fwd_flow_vis=visualize_flow_to_rgb(forward_flow, vector_stride=20)
    bwd_flow_vis=visualize_flow_to_rgb(backward_flow, vector_stride=20)
     # 保存光流可视化图像
    save_image(fwd_flow_vis, os.path.join(output_dir, "03_forward_flow_visualization.png"))
    save_image(bwd_flow_vis, os.path.join(output_dir, "04_backward_flow_visualization.png"))
    print(" -> 光流可视化图像已保存。")

    print("\n[步骤 5] 正在执行 Warp 操作并计算误差...")
    
    warp_s_bwd = flow_warp(source_tensor, backward_flow)
    mae_s_bwd = torch.mean(torch.abs(warp_s_bwd - target_tensor)).item()
    diff_s_bwd = torch.abs(warp_s_bwd - target_tensor)
    save_image(warp_s_bwd * 0.5 + 0.5, os.path.join(output_dir, "05_CORRECT_warp(source, bwd_flow).png"))
    save_image(diff_s_bwd, os.path.join(output_dir, "06_diff_vs_target.png"))
    print("Debug:",f"source tensor mean={torch.mean(source_tensor).item():.6f}, target tensor mean={torch.mean(target_tensor).item():.6f},warp_s_bwd mean={torch.mean(warp_s_bwd).item():.6f},forward flow mean={torch.mean(forward_flow).item():.6f}, backward flow mean={torch.mean(backward_flow).item():.6f}")
    print(f"  - 正确用法 1 (Warp(S, Bwd)): MAE vs Target = {mae_s_bwd:.6f}")

    warp_t_fwd = flow_warp(target_tensor, forward_flow)
    mae_t_fwd = torch.mean(torch.abs(warp_t_fwd - source_tensor)).item()
    diff_t_fwd = torch.abs(warp_t_fwd - source_tensor)
    save_image(warp_t_fwd * 0.5 + 0.5, os.path.join(output_dir, "07_CORRECT_warp(target, fwd_flow).png"))
    save_image(diff_t_fwd, os.path.join(output_dir, "08_diff_vs_source.png"))
    print(f"  - 正确用法 2 (Warp(T, Fwd)): MAE vs Source = {mae_t_fwd:.6f}")
    
    print(f"\n--- 结论 ---")
    print("脚本已运行完毕。")
    print(f"所有结果已保存至目录: {output_dir}")
    if mae_s_bwd < 0.1 and mae_t_fwd < 0.1:
         print("\n**好消息！** Warp 操作的误差很低，表明光流计算和Warp函数工作正常。")
    else:
         print("\n**注意！** Warp 误差较高。这可能是由于大的遮挡、快速运动或光照变化导致的。")


def extract_first_frame(video_path, output_image_path):
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 '{video_path}' 不存在。")
        return False
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
        cv2.imwrite(output_image_path, frame)
        print(f"已将视频第一帧保存至 '{output_image_path}'")
        return True
    else:
        print("错误: 无法从视频中读取第一帧。")
        return False

def load_pngs_as_tensors(source_path: str, target_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从指定的路径加载源图像和目标图像 (PNG)，并将其转换为 [-1, 1] 范围的浮点型Tensor。

    Args:
        source_path (str): 源图像文件 (source.png) 的路径。
        target_path (str): 目标图像文件 (target.png) 的路径。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 返回 (source_tensor, target_tensor)，
                                           形状均为 [1, 3, H, W]。
    """
    tensors = []
    for path in [source_path, target_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"图片文件未找到: {path}")
        
        # 使用 OpenCV 读取图像，默认格式为 BGR
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            raise IOError(f"无法读取图片: {path}")

        # 将 BGR 转换为 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 将图片从 0-255 (uint8) 转换为 -1 到 1 (float32) 的 Tensor
        # 流程: HWC (Numpy) -> CHW (Tensor) -> Add Batch dim -> Normalize
        image_tensor = (torch.from_numpy(image_rgb.astype(np.float32)).permute(2, 0, 1) / 127.5) - 1.0
        tensors.append(image_tensor.unsqueeze(0))

    return tensors[0], tensors[1]

if __name__ == "__main__":
    # --- 使用 argparse 处理命令行参数 ---
    parser = argparse.ArgumentParser(description="光流 Warp 操作验证脚本 (采用对齐缩放)")
    parser.add_argument("--source", type=str, default="./source.png", help="源图像文件路径 (source.png)")
    parser.add_argument("--target", type=str, default="./target.png", help="目标图像文件路径 (target.png)")
    parser.add_argument("--output_dir", type=str, default="test_visual_from_pngs", help="保存结果的输出目录")
    parser.add_argument("--height", type=int, default=480, help="目标图像高度 (将被对齐到32的倍数)")
    parser.add_argument("--width", type=int, default=832, help="目标图像宽度 (将被对齐到32的倍数)")
    
    args = parser.parse_args()

    try:
        # 从PNG文件加载图像为Tensor
        print(f"正在从 '{args.source}' 和 '{args.target}' 加载图像...")
        source_tensor, target_tensor = load_pngs_as_tensors(
            source_path=args.source, 
            target_path=args.target
        )
        print(" -> 图像加载成功。")

        # --- 新的缩放逻辑：与参考脚本保持一致 ---
        # 1. 定义对齐基数
        ALIGNMENT = 32
        # 2. 计算对齐后的目标尺寸 (向下取整到最近的32倍数)
        resize_hw = ((args.height // ALIGNMENT) * ALIGNMENT, (args.width // ALIGNMENT) * ALIGNMENT)

        # 3. 获取原始尺寸用于比较和日志记录
        _, _, H, W = source_tensor.shape
        
        # 4. 检查是否需要调整尺寸
        if H != resize_hw[0] or W != resize_hw[1]:
            print(f"检测到尺寸不匹配或需要对齐。正在将图像从 {W}x{H} 调整为 {resize_hw[1]}x{resize_hw[0]}...")
            # 5. 使用 torchvision.transforms.functional.resize 进行高质量缩放
            #    antialias=True 对于降采样时防止锯齿很重要
            source_tensor = TF.resize(source_tensor, resize_hw, antialias=True)
            target_tensor = TF.resize(target_tensor, resize_hw, antialias=True)
            print(" -> 图像尺寸调整完成。")
        else:
            print(f"图像尺寸 ({W}x{H}) 与目标对齐尺寸匹配，无需调整。")

        # 运行验证和保存流程
        run_verification_and_save_images(
            source_tensor=source_tensor,
            target_tensor=target_tensor,
            output_dir=args.output_dir
        )

    except (FileNotFoundError, IOError) as e:
        print(f"\n*** 运行错误 ***: {e}")
        print("请确保源图像和目标图像文件存在于正确的路径下。")
    except Exception as e:
        print(f"\n*** 发生未知错误 ***: {e}")
        import traceback
        traceback.print_exc()