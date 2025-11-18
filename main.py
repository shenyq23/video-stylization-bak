import os
import torch
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import argparse

from optical_wrapper import universal_flow_warp

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
    return img

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
            raise ValueError("failed to read frame")
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

def main(args):
    with torch.no_grad():
        video_path = os.path.join("./input", args.video_name, "input.mp4")
        stylized_video_path = os.path.join("./input", args.video_name, "stylized.mp4")
        width, height, fps, frames = load_video_frames(video_path=video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        stylized_width, stylized_height, stylized_fps, stylized_frames = load_video_frames(video_path=stylized_video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        assert len(frames) == len(stylized_frames), f"number of frames mismatch: {len(frames)} vs {len(stylized_frames)}"

        # resize the frames
        frames = [resize_image(frame, args.resolution) for frame in frames]
        stylized_frames = [resize_image(frame, args.resolution) for frame in stylized_frames]

        if args.flow_model == "gmflow":
            from optical_wrapper import GMFlowWrapper as FlowModelClass
        elif args.flow_model == "x265":
            from optical_wrapper import X265MVWrapper as FlowModelClass
        else:
            raise NotImplementedError(f"flow model {args.flow_model} not implemented.")
        
        flow_model = FlowModelClass(args.device)
        ref_frame_idx_list = [0] + [(i - 1) // args.batch_size * args.batch_size for i in range(1, len(frames))]
        flows, occlusions = flow_model.compute_flow_and_occlusion(frames, ref_frame_idx_list, size=f"{width}x{height}", frame_rate=fps)
        exit()

        key_frame_set = set(ref_frame_idx_list)
        forward_flows = flows[0]
        backward_flows = flows[1]
        backward_occlusions = occlusions[1]
        output_frames = [stylized_frames[0]]

        frame_process_timestamps = [time.time()]
        for i in range(1, len(frames)):
            if (not args.discard_key_frames) and i in key_frame_set:
                output_frames.append(stylized_frames[i])
                frame_process_timestamps.append(frame_process_timestamps[-1])
                print(f"frame #{i} is a key frame, skipped warping.")
                continue

            idx = ref_frame_idx_list[i]
            flow = backward_flows[i : i + 1]
            occlusion = backward_occlusions[i : i + 1]

            frame_tensor = numpy2tensor(np.array(stylized_frames[idx]), args.device)
            warped_frame = universal_flow_warp(frame_tensor, flow).squeeze_() * (1 - occlusion.squeeze_())
            output_frames.append(((warped_frame.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

            frame_process_timestamps.append(time.time())
            print(f"processed frame #{i} using flow from frame #{idx}. time elapsed: {1000 * (frame_process_timestamps[-1] - frame_process_timestamps[-2]):.4f} miliseconds.")

        frame_process_timestamps = np.array(frame_process_timestamps)
        time_diffs = frame_process_timestamps[1:] - frame_process_timestamps[:-1]
        print(f"average time per frame: {1000 * np.mean(time_diffs[time_diffs != 0]):.4f} miliseconds.")

    # pixel video
    final_output_frames = []
    output_path = os.path.join("./output", args.video_name, f"{args.flow_model}_batch_{args.batch_size}.mp4")
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

    # flow & occlusion video
    final_output_frames = []
    flow_output_path = os.path.join("./output", args.video_name, f"{args.flow_model}_flows_batch_{args.batch_size}.mp4")
    flow_frames = get_flow_frames(forward_flows)
    backward_occlusions_np = backward_occlusions.squeeze().cpu().numpy() * 255.0
    for i in range(len(flow_frames)):
        concat_frame = np.concatenate((flow_frames[i], backward_occlusions_np[i]), axis=1)
        final_output_frames.append(concat_frame.astype(np.uint8))
    with imageio.get_writer(flow_output_path, fps=args.frame_rate, codec="libx264") as writer:
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
    plot_output_path = os.path.join("./output", args.video_name, f"{args.flow_model}_occlusion_ratio_batch_{args.batch_size}.png")
    plt.savefig(plot_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow Video Processing")
    parser.add_argument("--video_name", type=str, help="Name of the video folder in ./input")
    parser.add_argument("--flow_model", type=str, choices=["gmflow", "x265"], help="Optical flow model to use")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing frames")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution to resize frames")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Starting frame index to process")
    parser.add_argument("--max_frames", type=int, default=40, help="Maximum number of frames to process")
    parser.add_argument("--frame_rate", type=int, default=8, help="Frame rate for output video")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--discard_key_frames", type=bool, default=False, help="Whether to keep key frames unchanged")

    args = parser.parse_args()
    main(args)
