import os
import gc
import torch
import numpy as np
from matplotlib import pyplot as plt

from optical_wrapper import GMFlowWrapper as FlowModelClass
from main import load_video_frames, resize_image

def main(resolution, frames, stylized_frames):
    with torch.no_grad():
        frames = [resize_image(frame, resolution) for frame in frames]
        stylized_frames = [resize_image(frame, resolution) for frame in stylized_frames]
        frames = [frame[0] for frame in frames]
        stylized_frames = [frame[0] for frame in stylized_frames]
        size_tuple = frames[0][1]
        size = f"{size_tuple[1]}x{size_tuple[0]}"
        
        flow_model = FlowModelClass("cuda:2")
        ref_frame_idx_list = [0] + [(i - 1) // 4 * 4 for i in range(1, len(frames))]
        flows, occlusions, time = flow_model.compute_flow_and_occlusion(frames, ref_frame_idx_list, size=size, frame_rate=fps)

        gc.collect()
        torch.cuda.empty_cache()
    return time

if __name__ == "__main__":
    video_path = os.path.join("./input", "road", "input.mp4")
    stylized_video_path = os.path.join("./input", "road", "stylized.mp4")
    width, height, fps, frames = load_video_frames(video_path=video_path, max_frames=100)
    stylized_width, stylized_height, stylized_fps, stylized_frames = load_video_frames(video_path=stylized_video_path, max_frames=100)
    assert len(frames) == len(stylized_frames), f"number of frames mismatch: {len(frames)} vs {len(stylized_frames)}"

    resolutions = [1080, 720, 480, 360, 240, 120]
    resolutions = list(reversed(resolutions))
    time_list = []
    for res in resolutions:
        sub_time_list = []
        for frame, stylized_frame in zip(frames, stylized_frames):
            time = main(res, [frame], [stylized_frame])
            sub_time_list.append(time)
        avg_time = np.mean(sub_time_list[3:])
        time_list.append(avg_time)

    with open("flow_time_vs_resolution.txt", "w") as f:
        for res, t in zip(resolutions, time_list):
            f.write(f"{res}p\t{t:4f}ms\n")

    plt.figure()
    plt.plot(resolutions, time_list, marker='o')
    plt.xlabel("Resolution (px)")
    plt.ylabel("Average Time per Frame (ms)")
    plt.title("Optical Flow Computation Time vs Resolution")
    plt.grid(True)
    plt.savefig("flow_time_vs_resolution.png")
