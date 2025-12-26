# ~/work/vedit/calflow/main.py

import os
# import sys  # <--- 不再需要

# disable torch warning
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
# sys.path.append("..") # <--- 删掉这一行！

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import argparse
import json

# --- 这行代码现在可以直接工作，无需任何修改 ---
from utils.optical_wrapper import universal_flow_warp

def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype in (np.float32, np.float64):
        # tolerate either [0,1] or [0,255]
        mx = float(np.max(frame)) if frame.size else 0.0
        if mx <= 1.5:
            frame = frame * 255.0
        return np.clip(frame, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


def diff_uint8_frames(a, b, mode: str, amplify: float, valid_mask: np.ndarray):
    a = to_uint8_rgb(a)
    b = to_uint8_rgb(b)
    if a.shape != b.shape:
        raise ValueError(f"frame shape mismatch: {a.shape} vs {b.shape}")

    diff = cv2.absdiff(a, b)  # uint8 RGB
    if valid_mask.shape != diff.shape[:2]:
        raise ValueError(f"valid_mask shape mismatch: {valid_mask.shape} vs {diff.shape[:2]}")
    valid = valid_mask.astype(bool)

    denom = int(valid.sum()) * 3
    if denom <= 0:
        mae = 0.0
    else:
        mae = float(diff.astype(np.float32)[valid].sum() / denom)

    amplify = float(amplify)
    if amplify != 1.0:
        diff = np.clip(diff.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    diff[~valid] = 0
    
    if mode == "abs_rgb":
        return diff, mae

    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    if valid is not None:
        diff_gray[~valid] = 0
    if mode == "abs_gray":
        return cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2RGB), mae
    if mode == "heatmap":
        heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return heat, mae

    raise ValueError(f"unknown diff mode: {mode}")

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img, (H, W)

def numpy2tensor(frame, device):
    x = torch.from_numpy(frame.copy()).float().to(device) / 255.0 * 2.0 - 1.
    x = torch.stack([x], dim=0)
    return x.permute(0, 3, 1, 2)

def load_video_frames(video_path, max_frames=None, start_frame_idx=0):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    frame_cnt = 0
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    while frame_cnt < start_frame_idx:
        success = video_capture.grab()
        if not success:
            raise ValueError("failed to grab frame")
        frame_cnt += 1
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if max_frames is not None and frame_cnt >= max_frames + start_frame_idx:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames += [frame]
        frame_cnt += 1

    print(f"successfully grab {len(frames)} frames with {width}x{height} @ {fps}fps")
    return width, height, fps, frames

def get_flow_frames(flows, vector_stride=20):
    H, W = flows[0].shape[1:]
    flow_frames = [np.zeros((H, W))]
    for i in range(1, len(flows)):
        flow_i_tensor = flows[i : i + 1]
        flow_canvas = np.zeros((H, W))
        flow_np = flow_i_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]

        for y in range(vector_stride // 2, H, vector_stride):
            for x in range(vector_stride // 2, W, vector_stride):
                dx, dy = flow_np[y, x, :]
                start_point = (x, y)
                end_x = int(np.clip(round(x + dx), 0, W - 1))
                end_y = int(np.clip(round(y + dy), 0, H - 1))
                end_point = (end_x, end_y)
                cv2.arrowedLine(flow_canvas, start_point, end_point, 128, 1, tipLength=0.3)
        flow_frames.append(flow_canvas)
    
    return flow_frames

def config_str(batch_size, model_name, x265_params):
    result = model_name
    if model_name == "x265" or model_name == "mix" or model_name == "reverse_mix":
        for k, v in x265_params.items():
            result += f"_{k}_{v}"
    return result + f"_{batch_size}"

def main(args):
    # --- 重要: 相对路径 `./input` 和 `./output` 意味着您需要从 `calflow` 根目录运行此脚本 ---
    with torch.no_grad():
        video_path = os.path.join("./motion_compensation/input", args.video_name, "input.mp4")
        stylized_video_path = os.path.join("./motion_compensation/input", args.video_name, "stylized.mp4")
        width, height, fps, frames = load_video_frames(video_path=video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        stylized_width, stylized_height, stylized_fps, stylized_frames = load_video_frames(video_path=stylized_video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        # assert len(frames) == len(stylized_frames), f"number of frames mismatch: {len(frames)} vs {len(stylized_frames)}"
        # --- Unified Resizing Logic ---
        if height != stylized_height or width != stylized_width:
            print(f"Resolution mismatch detected. Resizing original video from {width}x{height} to match stylized video at {stylized_width}x{stylized_height}.")
            target_dims = (stylized_width, stylized_height) # (W, H) for cv2.resize
            frames = [cv2.resize(f, target_dims, interpolation=cv2.INTER_AREA) for f in frames]
            width, height = stylized_width, stylized_height

        # resize the frames
        if args.resolution is not None:
            frames = [resize_image(frame, args.resolution) for frame in frames]
            stylized_frames = [resize_image(frame, args.resolution) for frame in stylized_frames]
            size_tuple = frames[0][1]
            size = f"{size_tuple[1]}x{size_tuple[0]}"
            frames = [frame[0] for frame in frames]
            stylized_frames = [frame[0] for frame in stylized_frames]
        else:
            assert width == stylized_width and height == stylized_height, "resolution mismatch between original and stylized videos"
            size = f"{width}x{height}"

        x265_params = {}
        # --- 这些导入现在可以直接工作 ---
        if args.flow_model == "gmflow":
            from utils.optical_wrapper import GMFlowWrapper as FlowModelClass
        elif args.flow_model == "raft":
            from utils.optical_wrapper import RAFTFlowWrapper as FlowModelClass
        elif args.flow_model == "x265":
            from utils.optical_wrapper import X265MVWrapper as FlowModelClass
            x265_params = json.loads(args.x265_params.replace("'", '"'))  # @todo: bad implementation
        elif args.flow_model == "mix":
            from utils.optical_wrapper import GMFlowWrapper as OcclusionModelClass
            from utils.optical_wrapper import X265MVWrapper as FlowModelClass
            x265_params = json.loads(args.x265_params.replace("'", '"'))  # @todo: bad implementation
        elif args.flow_model == "reverse_mix":
            from utils.optical_wrapper import X265MVWrapper as OcclusionModelClass
            from utils.optical_wrapper import GMFlowWrapper as FlowModelClass
            x265_params = json.loads(args.x265_params.replace("'", '"'))  # @todo: bad implementation
        else:
            raise NotImplementedError(f"flow model {args.flow_model} not implemented.")
        
        if "size" in x265_params:
            del x265_params["size"]
        if "frame_rate" in x265_params:
            del x265_params["frame_rate"]
        
        flow_model = FlowModelClass(args.device)

        # --- OPTIMIZED LOGIC FOR ref_frame_idx_list AS PER YOUR REQUEST ---
        # Rule: Frames are grouped by chunk_size. Chunk N references the last frame of chunk N-4.
        # If chunk N-4 does not exist (i.e., for the first 4 chunks), it references frame 0.
        chunk_size = args.batch_size
        ref_frame_idx_list = [max(0, (i // chunk_size - 3) * chunk_size - 1) if chunk_size > 0 else 0 for i in range(len(frames))]
        print(f"Generated reference frames using chunk_size={chunk_size}. Rule: chunk N refs last frame of chunk N-4.")
        print(f"Example reference indices: {ref_frame_idx_list[:20]}")
        
        flows, occlusions, _ = flow_model.compute_flow_and_occlusion(frames, ref_frame_idx_list, size=size, frame_rate=fps, **x265_params)

        if args.flow_model == "mix" or args.flow_model == "reverse_mix":
            occlusion_model = OcclusionModelClass(args.device)

            # overwrite occlusions using specialized occlusion model
            _, occlusions, _ = occlusion_model.compute_flow_and_occlusion(frames, ref_frame_idx_list, size=size, frame_rate=fps, **x265_params)

        # --- OPTIMIZED FRAME GENERATION LOOP ---
        # The key_frame_set and the conditional skip have been removed.
        # Every frame from 1 to N will now be generated by warping its reference frame.
        
        forward_flows = flows[0]
        backward_flows = flows[1]
        backward_occlusions = occlusions[1]
        
        # ==================== MODIFICATION START ====================
        
        output_frames = [stylized_frames[0]] # Frame 0 is the anchor, it is not warped.
        print(f"Processing Frame #0: Is an anchor frame, not warped.") # <--- 新增的打印语句

        for i in range(1, len(frames)):
            idx = ref_frame_idx_list[i]
            print(f"Processing Frame #{i}: Warping from reference Frame #{idx}") # <--- 新增的打印语句
            
            flow = backward_flows[i : i + 1]
            occlusion = backward_occlusions[i : i + 1]

            frame_tensor = numpy2tensor(np.array(stylized_frames[idx]), args.device)
            warped_frame = universal_flow_warp(frame_tensor, flow).squeeze_() * (1 - occlusion.squeeze_())
            output_frames.append(((warped_frame.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
            
        # ===================== MODIFICATION END =====================

    # config name
    config = config_str(args.batch_size, args.flow_model, x265_params)

    # pixel video
    final_output_frames = []
    output_path = os.path.join("./motion_compensation/output", args.video_name, config, "pixel.mp4")
    for i in range(len(frames)):
        concat_frame = np.concatenate((stylized_frames[i], output_frames[i], frames[i]), axis=1)
        final_output_frames.append(concat_frame)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with imageio.get_writer(output_path, fps=args.frame_rate, codec="libx264") as writer:
        for frame in final_output_frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

    # diff video (stylized vs output)
    if args.write_diff:
        diff_output_path = os.path.join("./motion_compensation/output", args.video_name, config, f"diff_{args.diff_mode}_amp_{args.diff_amplify}.mp4")
        diff_frames = []
        diff_mae = []
        for i in range(len(frames)):
            diff_frame, mae = diff_uint8_frames(
                stylized_frames[i],
                output_frames[i],
                args.diff_mode,
                args.diff_amplify,
                valid_mask=backward_occlusions[i].detach().squeeze().cpu().numpy(),
            )
            diff_frames.append(diff_frame)
            diff_mae.append(mae)
        with imageio.get_writer(diff_output_path, fps=args.frame_rate, codec="libx264") as writer:
            for frame in diff_frames:
                writer.append_data(frame)

        plt.figure()
        plt.plot(range(len(diff_mae)), diff_mae)
        plt.xlabel("Frame Index")
        plt.ylabel("Mean Absolute Pixel Difference")
        plt.title("Stylized vs Output per-frame diff")
        diff_plot_path = os.path.join("./motion_compensation/output", args.video_name, config, f"diff_{args.diff_mode}_amp_{args.diff_amplify}.png")
        plt.savefig(diff_plot_path)

    # flow & occlusion video
    final_output_frames = []
    flow_output_path = os.path.join("./motion_compensation/output", args.video_name, config, "flows.mp4")
    forward_flow_frames = get_flow_frames(forward_flows)
    backward_flow_frames = get_flow_frames(backward_flows)
    backward_occlusions_np = backward_occlusions.squeeze().cpu().numpy() * 255.0
    for i in range(len(forward_flow_frames)):
        concat_frame = np.concatenate((forward_flow_frames[i], backward_flow_frames[i], backward_occlusions_np[i]), axis=1)
        final_output_frames.append(concat_frame.astype(np.uint8))
    with imageio.get_writer(flow_output_path, fps=2, codec="libx264") as writer:
        for frame in final_output_frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

    # mask ratio plot
    occlusion_ratios = []
    for i in range(backward_occlusions.shape[0]):
        occlusion = backward_occlusions[i : i + 1]
        occlusion_np = occlusion.squeeze().cpu().numpy()
        occlusion_ratio = np.sum(occlusion_np) / (occlusion_np.shape[0] * occlusion_np.shape[1])
        occlusion_ratios.append(occlusion_ratio)
    plt.figure()
    plt.plot(range(len(occlusion_ratios)), occlusion_ratios)
    plt.xlabel("Frame Index")
    plt.ylabel("Occlusion Ratio")
    plt.title("Occlusion Ratio over Frames")
    plot_output_path = os.path.join("./motion_compensation/output", args.video_name, config, "occlusion_ratio.png")
    plt.savefig(plot_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow Video Processing")
    parser.add_argument("--video_name", type=str, help="Name of the video folder in ./motion_compensation/input")
    parser.add_argument("--flow_model", type=str, choices=["gmflow", "raft", "x265", "mix", "reverse_mix"], help="Optical flow model to use")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing frames")
    parser.add_argument("--resolution", type=int, default=None, help="Resolution to resize frames")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Starting frame index to process")
    parser.add_argument("--max_frames", type=int, default=40, help="Maximum number of frames to process")
    parser.add_argument("--frame_rate", type=int, default=8, help="Frame rate for output video")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--discard_key_frames", action="store_true", help="Whether to keep key frames unchanged")
    parser.add_argument("--x265_params", type=str, default="", help="Additional x265 params")
    parser.add_argument("--write_diff", action="store_true", help="Write a per-frame difference visualization between stylized and output")
    parser.add_argument("--diff_mode", type=str, default="heatmap", choices=["abs_rgb", "abs_gray", "heatmap"], help="Diff visualization mode")
    parser.add_argument("--diff_amplify", type=float, default=4.0, help="Amplify factor for diff (e.g., 4.0 makes small differences more visible)")

    args = parser.parse_args()
    main(args)