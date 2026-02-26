import os
import sys

# disable torch warning
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# Add at the BEGINNING of sys.path
sys.path.insert(0, "..")

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import argparse
import json

from utils.optical_wrapper import universal_flow_warp, OcclusionComputation
from utils.video_utils import (
    load_video_frames,
    resize_image,
    numpy2tensor,
    get_flow_frames,
    diff_uint8_frames,
)

def parse_x265_params(param_str):
    """
    1. KV pairs: "preset=fast stage=encode ctu=16"
    2. JSON format: "{'preset': 'fast', 'stage': 'encode'}"
    """
    if not param_str or param_str.strip() == "":
        return {}

    param_str = param_str.strip()

    # Check if it's JSON format (starts with { or ")
    if param_str.startswith('{') or param_str.startswith('"'):
        # Legacy JSON format
        return json.loads(param_str.replace("'", '"'))

    # Parse key=value format
    params = {}
    pairs = param_str.split()
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid x265 parameter format: '{pair}'. Expected 'key=value'")
        key, value = pair.split('=', 1)

        # Auto-convert types
        if value.lower() == 'true':
            params[key] = True
        elif value.lower() == 'false':
            params[key] = False
        elif value.isdigit():
            params[key] = int(value)
        elif value.replace('.', '', 1).isdigit():
            params[key] = float(value)
        else:
            params[key] = value

    return params

def config_str(batch_size, model_name, x265_params, occlusion_methods):
    result = model_name
    if model_name == "x265" or model_name == "mix" or model_name == "reverse_mix":
        for k, v in x265_params.items():
            result += f"_{k}_{v}"
    result += f"_{batch_size}"

    # Add occlusion methods to config name
    if occlusion_methods:
        occ_str = "occ_" + "_".join(sorted(occlusion_methods))
        result += f"_{occ_str}"

    return result

def main(args):
    with torch.no_grad():
        video_path = os.path.join("./input", args.video_name, "input.mp4")
        stylized_video_path = os.path.join("./input", args.video_name, "stylized.mp4")
        width, height, fps, frames = load_video_frames(video_path=video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        stylized_width, stylized_height, stylized_fps, stylized_frames = load_video_frames(video_path=stylized_video_path, max_frames=args.max_frames, start_frame_idx=args.start_frame_idx)
        assert len(frames) == len(stylized_frames), f"number of frames mismatch: {len(frames)} vs {len(stylized_frames)}"

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
        occlusion_flow_model = None  # For mix/reverse_mix modes

        if args.flow_model == "gmflow":
            from utils.optical_wrapper import GMFlowWrapper as FlowModelClass
        elif args.flow_model == "raft":
            from utils.optical_wrapper import RAFTFlowWrapper as FlowModelClass
        elif args.flow_model == "x265":
            from utils.optical_wrapper import X265MVWrapper as FlowModelClass
            x265_params = parse_x265_params(args.x265_params)
        elif args.flow_model == "mix":
            # mix: warping uses x265 flow, occlusion uses gmflow
            from utils.optical_wrapper import X265MVWrapper as FlowModelClass
            from utils.optical_wrapper import GMFlowWrapper as OcclusionFlowModelClass
            x265_params = parse_x265_params(args.x265_params)
        elif args.flow_model == "reverse_mix":
            # reverse_mix: warping uses gmflow, occlusion uses x265 flow
            from utils.optical_wrapper import GMFlowWrapper as FlowModelClass
            from utils.optical_wrapper import X265MVWrapper as OcclusionFlowModelClass
            x265_params = parse_x265_params(args.x265_params)
        else:
            raise NotImplementedError(f"flow model {args.flow_model} not implemented.")

        if "size" in x265_params:
            del x265_params["size"]
        if "frame_rate" in x265_params:
            del x265_params["frame_rate"]

        # Determine occlusion methods early for config name
        occlusion_methods = []
        if args.use_geometry: occlusion_methods.append('geometry')
        if args.use_luminosity: occlusion_methods.append('luminosity')
        if args.use_color: occlusion_methods.append('color')
        if args.use_structure: occlusion_methods.append('structure')

        if not occlusion_methods:
            raise ValueError("At least one occlusion method must be enabled")

        # Check if output already exists (before expensive computation)
        video_name = f"{args.video_name}-{size}"
        config = config_str(args.batch_size, args.flow_model, x265_params, occlusion_methods)
        output_dir = os.path.join("./output", video_name, config)
        if os.path.exists(output_dir):
            print(f"Output directory already exists: {output_dir}")
            print("Skipping computation. Use a different config or delete the directory to recompute.")
            return

        native_x265 = getattr(args, 'native_x265', False)
        reuse_x265 = bool(getattr(args, 'reuse_x265', 1))

        if args.flow_model == "x265" or args.flow_model == "mix":
            flow_model = FlowModelClass(args.device, native_x265=native_x265, reuse_x265_encoder=reuse_x265)
        else:
            flow_model = FlowModelClass(args.device)

        ref_frame_idx_list = [0] + [(i - 1) // args.batch_size * args.batch_size for i in range(1, len(frames))]

        # Compute optical flow for warping
        if args.flow_model == "x265" or args.flow_model == "mix":
            flows, _ = flow_model.compute_flow(frames, ref_frame_idx_list, size=size, frame_rate=fps, **x265_params)
            # flows, _ = flow_model.compute_flow(frames, ref_frame_idx_list, size=size, frame_rate=fps, **x265_params)
            # Print per-frame flow statistics for comparison
            forward_flows_array = flows[0].cpu().numpy() if torch.is_tensor(flows[0]) else flows[0]
            backward_flows_array = flows[1].cpu().numpy() if torch.is_tensor(flows[1]) else flows[1]
            print(f"\n{'='*80}")
            print(f"Flow Statistics (shape: {forward_flows_array.shape} = num_frames × channels × H × W)")
            print(f"{'='*80}")
            for i, ref_idx in enumerate(ref_frame_idx_list):
                # Flow direction explanation:
                # - Backward flow: motion from reference frame to current frame (ref → current)
                # - Forward flow: motion from current frame back to reference (current → ref)
                print(f"\n[Output Flow Index {i}]  Current=POC{i}, Reference=POC{ref_idx}")
 
                # Backward flow: POC{ref_idx} → POC{i}
                bwd_flow = backward_flows_array[i]
                bwd_min, bwd_max = bwd_flow.min(), bwd_flow.max()
                bwd_mean, bwd_std = bwd_flow.mean(), bwd_flow.std()
                bwd_nonzero = np.count_nonzero(bwd_flow)
                bwd_total = bwd_flow.size
                bwd_ratio = bwd_nonzero / bwd_total * 100
                print(f"  Bwd  (POC{ref_idx}→POC{i}):  range=[{bwd_min:8.4f}, {bwd_max:8.4f}]  "
                      f"mean={bwd_mean:8.4f}  std={bwd_std:7.4f}  "
                      f"nonzero={bwd_nonzero:7d}/{bwd_total:7d} ({bwd_ratio:5.1f}%)")

                # Forward flow: POC{i} → POC{ref_idx}
                fwd_flow = forward_flows_array[i]
                fwd_min, fwd_max = fwd_flow.min(), fwd_flow.max()
                fwd_mean, fwd_std = fwd_flow.mean(), fwd_flow.std()
                fwd_nonzero = np.count_nonzero(fwd_flow)
                fwd_total = fwd_flow.size
                fwd_ratio = fwd_nonzero / fwd_total * 100
                print(f"  Fwd  (POC{i}→POC{ref_idx}):  range=[{fwd_min:8.4f}, {fwd_max:8.4f}]  "
                      f"mean={fwd_mean:8.4f}  std={fwd_std:7.4f}  "
                      f"nonzero={fwd_nonzero:7d}/{fwd_total:7d} ({fwd_ratio:5.1f}%)")
            print(f"{'='*80}\n")
        else:
            flows, _ = flow_model.compute_flow(frames, ref_frame_idx_list)

        # Compute optical flow for occlusion (if using mix/reverse_mix)
        if args.flow_model == "mix" or args.flow_model == "reverse_mix":
            if args.flow_model == "reverse_mix":
                occlusion_flow_model = OcclusionFlowModelClass(args.device, native_x265=native_x265, reuse_x265_encoder=reuse_x265)
            else:
                occlusion_flow_model = OcclusionFlowModelClass(args.device)
            if args.flow_model == "reverse_mix":
                # reverse_mix: occlusion uses x265
                occlusion_flows, _ = occlusion_flow_model.compute_flow(frames, ref_frame_idx_list, size=size, frame_rate=fps, **x265_params)
            else:
                # mix: occlusion uses gmflow
                occlusion_flows, _ = occlusion_flow_model.compute_flow(frames, ref_frame_idx_list)
        else:
            # For non-mix modes, use the same flow for both warping and occlusion
            occlusion_flows = flows

        # Convert frames to tensors for occlusion computation
        images = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames], dim=0).to(args.device)

        # Compute occlusion using OcclusionComputation
        occlusion_computer = OcclusionComputation(
            use_geometry=args.use_geometry,
            use_luminosity=args.use_luminosity,
            use_color=args.use_color,
            use_structure=args.use_structure,
            combine_method=args.occlusion_combine_method,
            geometry_threshold=(args.geometry_threshold_alpha, args.geometry_threshold_beta),
            luminosity_threshold=args.luminosity_threshold,
            color_threshold=args.color_threshold,
            structure_threshold=args.structure_threshold,
        )

        # Use warping flows for final output
        forward_flows, backward_flows = flows
        # Use occlusion flows for occlusion computation
        occlusion_forward_flows, occlusion_backward_flows = occlusion_flows

        forward_occlusions = torch.zeros((len(frames), images.shape[2], images.shape[3]), device=args.device, dtype=torch.float32)
        backward_occlusions = torch.zeros((len(frames), images.shape[2], images.shape[3]), device=args.device, dtype=torch.float32)

        for i, ref_idx in enumerate(ref_frame_idx_list):
            src_frame = images[ref_idx : ref_idx + 1]
            tgt_frame = images[i : i + 1]
            forward_flow = occlusion_forward_flows[i : i + 1]
            backward_flow = occlusion_backward_flows[i : i + 1]

            forward_occ, backward_occ = occlusion_computer(src_frame, tgt_frame, forward_flow, backward_flow)
            forward_occlusions[i] = forward_occ[0]
            backward_occlusions[i] = backward_occ[0]

        occlusions = [forward_occlusions, backward_occlusions]

        key_frame_set = set(ref_frame_idx_list)
        forward_flows = flows[0]
        backward_flows = flows[1]
        backward_occlusions = occlusions[1]
        output_frames = [stylized_frames[0]]

        for i in range(1, len(frames)):
            if (not args.discard_key_frames) and i in key_frame_set:
                output_frames.append(stylized_frames[i])
                print(f"frame #{i} is a key frame, skipped warping.")
                continue

            idx = ref_frame_idx_list[i]
            flow = backward_flows[i : i + 1]
            occlusion = backward_occlusions[i : i + 1]

            frame_tensor = numpy2tensor(np.array(stylized_frames[idx]), args.device)
            warped_frame = universal_flow_warp(frame_tensor, flow).squeeze_() * (1 - occlusion.squeeze_())
            output_frames.append(((warped_frame.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

    # pixel video
    final_output_frames = []
    output_path = os.path.join("./output", video_name, config, "pixel.mp4")
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
        diff_output_path = os.path.join("./output", video_name, config, f"diff_{args.diff_mode}_amp_{args.diff_amplify}.mp4")
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
        diff_plot_path = os.path.join("./output", video_name, config, f"diff_{args.diff_mode}_amp_{args.diff_amplify}.png")
        plt.savefig(diff_plot_path)

    # flow & occlusion video
    final_output_frames = []
    flow_output_path = os.path.join("./output", video_name, config, "flows.mp4")
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
    plot_output_path = os.path.join("./output", video_name, config, "occlusion_ratio.png")
    plt.savefig(plot_output_path)

def run_motion_compensation(**kwargs):
    class Args:
        pass
    args = Args()

    # Set defaults
    defaults = {
        "resolution": None,
        "start_frame_idx": 0,
        "max_frames": 40,
        "frame_rate": 8,
        "device": "cuda:0",
        "discard_key_frames": False,
        "x265_params": "",
        "use_geometry": False,
        "use_luminosity": False,
        "use_color": False,
        "use_structure": False,
        "geometry_threshold_alpha": 0.01,
        "geometry_threshold_beta": 0.5,
        "luminosity_threshold": 64,
        "color_threshold": 64,
        "structure_threshold": 50,
        "occlusion_combine_method": "mean",
        "write_diff": False,
        "diff_mode": "heatmap",
        "diff_amplify": 4.0,
        "native_x265": False,  # Use native zero-IO x265 mode
    }

    # Check required arguments
    required_args = ["video_name", "flow_model", "batch_size"]
    for req_arg in required_args:
        if req_arg not in kwargs:
            raise ValueError(f"Required argument '{req_arg}' is missing")

    # Validate flow_model
    valid_flow_models = ["gmflow", "raft", "x265", "mix", "reverse_mix"]
    if kwargs["flow_model"] not in valid_flow_models:
        raise ValueError(f"flow_model must be one of {valid_flow_models}")

    # Merge defaults with provided kwargs
    all_args = {**defaults, **kwargs}

    # Set attributes on args object
    for key, value in all_args.items():
        setattr(args, key, value)

    # Call main function
    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow Video Processing")
    parser.add_argument("--video_name", type=str, help="Name of the video folder in ./input")
    parser.add_argument("--flow_model", type=str, choices=["gmflow", "raft", "x265", "mix", "reverse_mix"], help="Optical flow model to use")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing frames")
    parser.add_argument("--resolution", type=int, default=None, help="Resolution to resize frames")
    parser.add_argument("--start_frame_idx", type=int, default=0, help="Starting frame index to process")
    parser.add_argument("--max_frames", type=int, default=40, help="Maximum number of frames to process")
    parser.add_argument("--frame_rate", type=int, default=8, help="Frame rate for output video")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--discard_key_frames", action="store_true", help="Whether to keep key frames unchanged")
    parser.add_argument("--x265_params", type=str, default="", help="X265 parameters in key=value format (e.g., 'preset=fast stage=encode ctu=16') or JSON format")
    parser.add_argument("--use_geometry", action="store_true", help="Use geometry-based occlusion detection (forward-backward consistency)")
    parser.add_argument("--use_luminosity", action="store_true", help="Use luminosity-based occlusion detection (photometric check with mean)")
    parser.add_argument("--use_color", action="store_true", help="Use color-based occlusion detection (per-channel photometric check)")
    parser.add_argument("--use_structure", action="store_true", help="Use structure-based occlusion detection (gradient-based check)")
    parser.add_argument("--geometry_threshold_alpha", type=float, default=0.01, help="Alpha parameter for geometry threshold (error > alpha * flow_mag + beta)")
    parser.add_argument("--geometry_threshold_beta", type=float, default=0.5, help="Beta parameter for geometry threshold (error > alpha * flow_mag + beta)")
    parser.add_argument("--luminosity_threshold", type=float, default=64, help="Threshold for luminosity-based occlusion detection")
    parser.add_argument("--color_threshold", type=float, default=64, help="Threshold for color-based occlusion detection")
    parser.add_argument("--structure_threshold", type=float, default=50, help="Threshold for structure-based occlusion detection")
    parser.add_argument("--occlusion_combine_method", type=str, default="mean", choices=["mean", "max", "sum"], help="Method to combine multiple occlusion components: mean (average), max (union), or sum (clamped sum)")
    parser.add_argument("--write_diff", action="store_true", help="Write a per-frame difference visualization between stylized and output")
    parser.add_argument("--diff_mode", type=str, default="heatmap", choices=["abs_rgb", "abs_gray", "heatmap"], help="Diff visualization mode")
    parser.add_argument("--diff_amplify", type=float, default=4.0, help="Amplify factor for diff (e.g., 4.0 makes small differences more visible)")
    parser.add_argument("--native_x265", action="store_true", help="Use native zero-IO x265 mode (eliminates file I/O for ~2.4x speedup)")
    parser.add_argument("--reuse_x265", type=int, default=1, help="Reuse x265 encoder via reset (default: 1=enabled, 0=disabled; native mode only, reduces encoder open/close overhead)")

    args = parser.parse_args()
    main(args)
