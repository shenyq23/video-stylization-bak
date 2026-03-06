import os
import sys
import time
import gc
import torch
import tempfile
import cv2
import pandas as pd
import numpy as np
from types import SimpleNamespace
from collections import OrderedDict

sys.path.append("../deps/gmflow")
# 为了使RAFT正常工作，请取消以下两行的注释
# sys.path.append("../deps/RAFT")
# from core.raft import RAFT

from utils.x265_wrapper import X265EncoderWrapper
from gmflow.gmflow import GMFlow
from gmflow.geometry import flow_warp

# disable RAFT
# from deps.RAFT.core.raft import RAFT
class RAFT:
    pass

# the warp function from gmflow can be used universally
def universal_flow_warp(frame, flow):
    return flow_warp(frame, flow)

class OcclusionComputation:
    def __init__(
        self,
        use_geometry=False,
        use_luminosity=False,
        use_color=False,
        use_structure=False,
        combine_method='mean',  # 'mean', 'max', or 'sum'
        geometry_threshold=(0.01, 0.5),
        luminosity_threshold=64,
        color_threshold=64,
        structure_threshold=50,  # temporarily these thresholds are fixed
    ):
        assert use_geometry or use_luminosity or use_color or use_structure
        if use_geometry: assert geometry_threshold is not None
        if use_luminosity: assert luminosity_threshold is not None
        if use_color: assert color_threshold is not None
        if use_structure: assert structure_threshold is not None
        assert combine_method in ['mean', 'max', 'sum'], f"Invalid combine_method: {combine_method}"
        self.use_geometry = use_geometry
        self.use_luminosity = use_luminosity
        self.use_color = use_color
        self.use_structure = use_structure
        self.combine_method = combine_method

        self.geometry_threshold = geometry_threshold
        self.luminosity_threshold = luminosity_threshold
        self.color_threshold = color_threshold
        self.structure_threshold = structure_threshold

    @staticmethod
    def geometry_error(src_frame, tgt_frame, forward_flow, backward_flow):
        # Compute continuous occlusion based on forward-backward consistency
        # Similar to forward_backward_consistency_check but returns continuous values
        warped_bwd_flow = flow_warp(backward_flow, forward_flow)  # [B, 2, H, W]
        warped_fwd_flow = flow_warp(forward_flow, backward_flow)  # [B, 2, H, W]

        diff_fwd = torch.norm(forward_flow + warped_bwd_flow, dim=1)  # [B, H, W]
        diff_bwd = torch.norm(backward_flow + warped_fwd_flow, dim=1)

        return diff_fwd, diff_bwd

    @staticmethod
    def luminosity_error(src_frame, tgt_frame, forward_flow, backward_flow):
        # Forward occlusion: warp target with forward flow, compare to source
        warped_target = flow_warp(tgt_frame, forward_flow)
        forward_photo_error = torch.abs(src_frame - warped_target).mean(dim=1)  # [N, H, W]

        # Backward occlusion: warp source with backward flow, compare to target
        warped_source = flow_warp(src_frame, backward_flow)
        backward_photo_error = torch.abs(tgt_frame - warped_source).mean(dim=1)  # [N, H, W]

        return forward_photo_error, backward_photo_error

    @staticmethod
    def color_error(src_frame, tgt_frame, forward_flow, backward_flow):
        # Forward occlusion: warp target with forward flow, compare to source
        warped_target = flow_warp(tgt_frame, forward_flow)
        forward_color_error = torch.abs(src_frame - warped_target)  # [N, 3, H, W]

        # Backward occlusion: warp source with backward flow, compare to target
        warped_source = flow_warp(src_frame, backward_flow)
        backward_color_error = torch.abs(tgt_frame - warped_source)  # [N, 3, H, W]

        # Use max across channels for color sensitivity
        forward_color_error = forward_color_error.max(dim=1)[0]  # [N, H, W]
        backward_color_error = backward_color_error.max(dim=1)[0]  # [N, H, W]

        return forward_color_error, backward_color_error

    @staticmethod
    def structure_error(src_frame, tgt_frame, forward_flow, backward_flow):
        def compute_gradients(img):
            # Compute image gradients using Sobel-like operators
            # img: [N, 3, H, W]
            dx_kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=img.dtype, device=img.device)
            dy_kernel = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=img.dtype, device=img.device)

            # Expand kernels for all channels
            dx_kernel = dx_kernel.repeat(3, 1, 1, 1) / 8.0  # [3, 1, 3, 3]
            dy_kernel = dy_kernel.repeat(3, 1, 1, 1) / 8.0  # [3, 1, 3, 3]

            # Compute gradients
            grad_x = torch.nn.functional.conv2d(img, dx_kernel, padding=1, groups=3)
            grad_y = torch.nn.functional.conv2d(img, dy_kernel, padding=1, groups=3)

            # Gradient magnitude
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            return grad_mag

        # Forward occlusion: compare gradients of source and warped target
        warped_target = flow_warp(tgt_frame, forward_flow)
        src_grad = compute_gradients(src_frame)
        warped_tgt_grad = compute_gradients(warped_target)
        forward_struct_error = torch.abs(src_grad - warped_tgt_grad).mean(dim=1)  # [N, H, W]

        # Backward occlusion: compare gradients of target and warped source
        warped_source = flow_warp(src_frame, backward_flow)
        tgt_grad = compute_gradients(tgt_frame)
        warped_src_grad = compute_gradients(warped_source)
        backward_struct_error = torch.abs(tgt_grad - warped_src_grad).mean(dim=1)  # [N, H, W]

        return forward_struct_error, backward_struct_error

    def __call__(self, src_frame, tgt_frame, forward_flow, backward_flow):
        forward_occlusion_components = []
        backward_occlusion_components = []

        if self.use_geometry:
            forward_error, backward_error = OcclusionComputation.geometry_error(src_frame, tgt_frame, forward_flow, backward_flow)
            # Geometry uses gmflow's method: error > alpha * flow_mag + beta
            # We convert this to continuous [0, 1] by normalizing with the threshold
            flow_mag = torch.norm(forward_flow, dim=1) + torch.norm(backward_flow, dim=1)  # [B, H, W]
            alpha, beta = self.geometry_threshold  # (alpha, beta) tuple
            threshold_fwd = alpha * flow_mag + beta
            threshold_bwd = alpha * flow_mag + beta
            forward_occ = torch.clamp(forward_error / (threshold_fwd + 1e-6), 0, 1)
            backward_occ = torch.clamp(backward_error / (threshold_bwd + 1e-6), 0, 1)
            forward_occlusion_components.append(forward_occ)
            backward_occlusion_components.append(backward_occ)

        if self.use_luminosity:
            forward_error, backward_error = OcclusionComputation.luminosity_error(src_frame, tgt_frame, forward_flow, backward_flow)
            # Luminosity: normalize by fixed threshold
            forward_occ = torch.clamp(forward_error / self.luminosity_threshold, 0, 1)
            backward_occ = torch.clamp(backward_error / self.luminosity_threshold, 0, 1)
            forward_occlusion_components.append(forward_occ)
            backward_occlusion_components.append(backward_occ)

        if self.use_color:
            forward_error, backward_error = OcclusionComputation.color_error(src_frame, tgt_frame, forward_flow, backward_flow)
            # Color: normalize by fixed threshold
            forward_occ = torch.clamp(forward_error / self.color_threshold, 0, 1)
            backward_occ = torch.clamp(backward_error / self.color_threshold, 0, 1)
            forward_occlusion_components.append(forward_occ)
            backward_occlusion_components.append(backward_occ)

        if self.use_structure:
            forward_error, backward_error = OcclusionComputation.structure_error(src_frame, tgt_frame, forward_flow, backward_flow)
            # Structure: normalize by fixed threshold
            forward_occ = torch.clamp(forward_error / self.structure_threshold, 0, 1)
            backward_occ = torch.clamp(backward_error / self.structure_threshold, 0, 1)
            forward_occlusion_components.append(forward_occ)
            backward_occlusion_components.append(backward_occ)

        # Combine occlusion components based on combine_method
        if len(forward_occlusion_components) > 0:
            forward_stack = torch.stack(forward_occlusion_components, dim=0)
            backward_stack = torch.stack(backward_occlusion_components, dim=0)

            if self.combine_method == 'mean':
                combined_forward_occlusion = forward_stack.mean(dim=0)
                combined_backward_occlusion = backward_stack.mean(dim=0)
            elif self.combine_method == 'max':
                combined_forward_occlusion = forward_stack.max(dim=0)[0]
                combined_backward_occlusion = backward_stack.max(dim=0)[0]
            elif self.combine_method == 'sum':
                combined_forward_occlusion = torch.clamp(forward_stack.sum(dim=0), 0, 1)
                combined_backward_occlusion = torch.clamp(backward_stack.sum(dim=0), 0, 1)
        else:
            # This shouldn't happen due to the assert in __init__
            combined_forward_occlusion = torch.zeros_like(forward_flow[:, 0])
            combined_backward_occlusion = torch.zeros_like(backward_flow[:, 0])

        return combined_forward_occlusion, combined_backward_occlusion

class OpticalFlowWrapper:
    def __init__(self, device):
        self.device = device

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        raise NotImplementedError

class GMFlowWrapper(OpticalFlowWrapper):
    def _load_gmflow_model(self):
        optical_flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(self.device)

        checkpoint = torch.load("../deps/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth", map_location=lambda storage, loc: storage)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        optical_flow_model.load_state_dict(weights, strict=False)
        optical_flow_model.eval()
        optical_flow_model.to(self.device)

        for param in optical_flow_model.parameters():
            param.requires_grad = False

        return optical_flow_model

    def __init__(self, device):
        super().__init__(device)
        self.model = self._load_gmflow_model()

    def compute_flow_from_tensors(self, ref_frame_tensor: torch.Tensor, current_frame_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes bidirectional optical flow and occlusion directly from PyTorch tensors.
        This method is optimized for pipelines where frames are already on the GPU as tensors.

        Args:
            ref_frame_tensor (torch.Tensor): The reference frame (source). 
                                             Expected shape [B, C, H, W] and range [-1, 1].
                                             The dtype and device of this tensor will determine the output's dtype and device.
            current_frame_tensor (torch.Tensor): The current frame (target).
                                                 Expected shape [B, C, H, W] and range [-1, 1].

        Returns:
            A tuple containing:
                - forward_flow (torch.Tensor): Flow from reference to current frame. Shape [B, 2, H, W].
                - backward_flow (torch.Tensor): Flow from current to reference frame. Shape [B, 2, H, W].
                - forward_occlusion (torch.Tensor): Occlusion mask for forward flow. Shape [B, 1, H, W].
                - backward_occlusion (torch.Tensor): Occlusion mask for backward flow. Shape [B, 1, H, W].
            All returned tensors will have the same dtype and device as the input `ref_frame_tensor`.
        """
        with torch.no_grad():
            # --- MODIFICATION START ---
            # Store original dtype and device to ensure output consistency.
            input_dtype = ref_frame_tensor.dtype
            input_device = ref_frame_tensor.device

            # Convert input tensors from [-1, 1] range to [0, 255] range for the GMFlow model.
            source_frame_255 = ((ref_frame_tensor * 0.5 + 0.5) * 255.0).float()
            target_frame_255 = ((current_frame_tensor * 0.5 + 0.5) * 255.0).float()
            # --- MODIFICATION END ---

            _, _, H, W = source_frame_255.shape
            feat_h = H // 8
            feat_w = W // 8
            attn_split = 2 if (feat_h % 2 == 0 and feat_w % 2 == 0) else 1

            # with torch.amp.autocast("cuda", enabled=True):
            results_dict = self.model(
                source_frame_255, # Use the converted tensor
                target_frame_255, # Use the converted tensor
                attn_splits_list=[attn_split],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=True,
            )
            flow_prediction = results_dict["flow_preds"][-1]
            forward_flow, backward_flow = flow_prediction.chunk(2)

            # --- MODIFICATION START ---
            # Convert all output tensors to the original input dtype and device before returning.
            return (
                forward_flow.to(dtype=input_dtype, device=input_device),
                backward_flow.to(dtype=input_dtype, device=input_device)
            )

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        """ no other kwargs needed for gmflow
        """
        with torch.no_grad():
            images = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames], dim=0).to(self.device)
            _, _, H, W = images.shape
            feat_h = H // 8
            feat_w = W // 8
            attn_split = 2 if (feat_h % 2 == 0 and feat_w % 2 == 0) else 1

            forward_flows = torch.zeros((len(frames), 2, H, W), device=self.device, dtype=torch.float32)
            backward_flows = torch.zeros((len(frames), 2, H, W), device=self.device, dtype=torch.float32)

            elapsed_times = []
            for i, ref_idx in enumerate(ref_frame_idx_list):
                source_frame = images[ref_idx : ref_idx + 1]
                target_frame = images[i : i + 1]

                with torch.amp.autocast("cuda:1", enabled=True):
                    start = time.time()
                    results_dict = self.model(
                        source_frame,
                        target_frame,
                        attn_splits_list=[attn_split],
                        corr_radius_list=[-1],
                        prop_radius_list=[-1],
                        pred_bidir_flow=True,
                    )
                    end = time.time()
                    elapsed_times.append(end - start)

                    flow_prediction = results_dict["flow_preds"][-1]  # [2, 2, H, W]
                    forward_flow, backward_flow = flow_prediction.chunk(2)

                forward_flows[i] = forward_flow[0].float()
                backward_flows[i] = backward_flow[0].float()

                gc.collect()
                torch.cuda.empty_cache()

                print(f"Processed frame #{i}/{len(frames)}: {elapsed_times[-1] * 1000:.4f} ms")

            return [forward_flows, backward_flows], np.mean(elapsed_times) * 1000

class RAFTFlowWrapper(OpticalFlowWrapper):
    def __init__(self, device):
        super().__init__(device)

        class RAFTArgs(SimpleNamespace):
            def __contains__(self, key):
                return hasattr(self, key)

        args = RAFTArgs()
        args.small = False
        args.mixed_precision = False
        args.dropout = 0
        args.alternate_corr = False

        # instantiate RAFT with args
        raft_model = RAFT(args)

        # wrap with DataParallel if multiple GPUs available (keeps consistency with previous code)
        if torch.cuda.device_count() > 1:
            raft_model = torch.nn.DataParallel(raft_model)

        # load checkpoint robustly (supports several common key formats)
        ckpt_path = "./deps/RAFT/models/raft-things.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state = ckpt['state_dict']
                elif 'model' in ckpt:
                    state = ckpt['model']
                else:
                    state = ckpt
            else:
                state = ckpt

            try:
                raft_model.load_state_dict(state, strict=False)
            except Exception:
                # try stripping possible 'module.' prefixes
                from collections import OrderedDict

                new_state = OrderedDict()
                for k, v in state.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state[name] = v
                raft_model.load_state_dict(new_state, strict=False)
        else:
            print(f"RAFT checkpoint not found at {ckpt_path}; initialized with random weights")

        self.model = raft_model.module if isinstance(raft_model, torch.nn.DataParallel) else raft_model
        self.model.to(self.device)
        self.model.eval()

    # --- NEW METHOD START ---
    def compute_flow_from_tensors(self, ref_frame_tensor: torch.Tensor, current_frame_tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes bidirectional optical flow using RAFT directly from PyTorch tensors.

        Args:
            ref_frame_tensor (torch.Tensor): Reference frame(s), shape [B, C, H, W], range [-1, 1].
            current_frame_tensor (torch.Tensor): Current frame(s), shape [B, C, H, W], range [-1, 1].
            **kwargs: Accepts 'iters' (int, default 12) for RAFT's refinement iterations.

        Returns:
            A tuple containing:
                - forward_flow (torch.Tensor): Flow from reference to current frame. Shape [B, 2, H, W].
                - backward_flow (torch.Tensor): Flow from current to reference frame. Shape [B, 2, H, W].
        """
        with torch.no_grad():
            input_dtype = ref_frame_tensor.dtype
            input_device = ref_frame_tensor.device
            iters = kwargs.get('iters', 12)

            # Convert input tensors from [-1, 1] range to [0, 255] range for the RAFT model.
            source_frame_255 = ((ref_frame_tensor * 0.5 + 0.5) * 255.0).float()
            target_frame_255 = ((current_frame_tensor * 0.5 + 0.5) * 255.0).float()

            # Forward flow: source -> target
            # The model returns a list of flow predictions; the last one is the final estimate.
            results_fwd = self.model(source_frame_255, target_frame_255, iters=iters, test_mode=True)
            forward_flow = results_fwd[-1] if isinstance(results_fwd, (list, tuple)) else results_fwd

            # Backward flow: target -> source
            results_bwd = self.model(target_frame_255, source_frame_255, iters=iters, test_mode=True)
            backward_flow = results_bwd[-1] if isinstance(results_bwd, (list, tuple)) else results_bwd

            return (
                forward_flow.to(dtype=input_dtype, device=input_device),
                backward_flow.to(dtype=input_dtype, device=input_device)
            )
    # --- NEW METHOD END ---

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        # frames: list/iterable of HxWx3 numpy arrays (BGR, 0-255)
        # ref_frame_idx_list: list mapping target index -> reference index
        with torch.no_grad():
            images = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames], dim=0).to(self.device)
            _, _, H, W = images.shape

            forward_flows = torch.zeros((len(frames), 2, H, W), device=self.device, dtype=torch.float32)
            backward_flows = torch.zeros((len(frames), 2, H, W), device=self.device, dtype=torch.float32)

            elapsed_times = []
            for i, ref_idx in enumerate(ref_frame_idx_list):
                source_frame = images[ref_idx : ref_idx + 1]
                target_frame = images[i : i + 1]

                # forward: source -> target
                with torch.cuda.amp.autocast(enabled=True):
                    start = time.time()
                    results_fwd = self.model(source_frame, target_frame)
                    end = time.time()
                    elapsed_times.append(end - start)

                    flow_pred_fwd = results_fwd[-1] if isinstance(results_fwd, (list, tuple)) else results_fwd

                # backward: target -> source
                with torch.cuda.amp.autocast(enabled=True):
                    start = time.time()
                    results_bwd = self.model(target_frame, source_frame)
                    end = time.time()
                    elapsed_times.append(end - start)

                    flow_pred_bwd = results_bwd[-1] if isinstance(results_bwd, (list, tuple)) else results_bwd

                # flow_pred_* : [N, 2, H, W]
                forward_flows[i] = flow_pred_fwd[0].float()
                backward_flows[i] = flow_pred_bwd[0].float()

                gc.collect()
                torch.cuda.empty_cache()

                print(f"RAFT: Processed frame #{i}/{len(frames)}: {elapsed_times[-1] * 1000:.4f} ms")

            return [forward_flows, backward_flows], np.mean(elapsed_times) * 1000

class X265MVWrapper(OpticalFlowWrapper):
    def __init__(self, device, encoder_path=None, native_x265=False, reuse_x265_encoder=True):
        """
            encoder_path: path to x265 encoder (for CSV mode)
            reuse_x265_encoder: if True, reuse encoder via x265_encoder_reset (native mode only)
        """
        super().__init__(device)
        self.native_x265 = native_x265
        self.reuse_x265_encoder = reuse_x265_encoder
        if native_x265:
            from utils.x265_native import X265NativeWrapper
            self.native_wrapper = X265NativeWrapper(device=device, reuse_encoder=reuse_x265_encoder)
        else:
            self.encoder = X265EncoderWrapper(encoder_path)

    @staticmethod
    def _update_flow(flow_log_path, flows_ref, ref_idx, granularity):
        df = pd.read_csv(flow_log_path)
        for _, row in df.iterrows():
            if int(row["poc"]) == 0:
                continue
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            mvx = float(row["mvx"])
            mvy = float(row["mvy"])
            delta_poc = int(row["deltapoc"])

            # x265 uses quarter pixel precision
            if "mv_precision" in df.columns:
                mv_precision = float(row["mv_precision"])
                mvx = mvx * mv_precision
                mvy = mvy * mv_precision

            assert w == granularity and h == granularity
            if delta_poc == 0:
                flows_ref[ref_idx, 0, y : y + h, x : x + w] = 0
                flows_ref[ref_idx, 1, y : y + h, x : x + w] = 0
            else:
                flows_ref[ref_idx, 0, y : y + h, x : x + w] = mvx
                flows_ref[ref_idx, 1, y : y + h, x : x + w] = mvy

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        """
            frames: List of BGR frames
            ref_frame_idx_list: Reference frame indices
            **kwargs: Encoding parameters
        """
        print(kwargs)
        # zero-IO native x265 mode first
        if self.native_x265:
            return self.native_wrapper.compute_flow(frames, ref_frame_idx_list, **kwargs)

        width = int(kwargs["size"].split("x")[0])
        height = int(kwargs["size"].split("x")[1])
        forward_flows = np.zeros((len(frames), 2, height, width), dtype=np.float32)
        backward_flows = np.zeros((len(frames), 2, height, width), dtype=np.float32)

        x265_params = {
            "preset": kwargs.get("preset", "fast"),
            "ctu": kwargs.get("ctu", 16),
            "crf": kwargs.get("crf", 23),
            "enable_p_intra": kwargs.get("enable_p_intra", False),
            "lookahead_intra": kwargs.get("lookahead_intra", False),
        }
        
        with tempfile.TemporaryDirectory() as tempdir:
            log_root = os.path.join(tempdir, "x265_logs")
            for idx, ref_idx in enumerate(ref_frame_idx_list):
                target_frame = frames[idx]
                source_frame = frames[ref_idx]

                print(source_frame.shape, target_frame.shape,forward_flows.shape, backward_flows.shape)

                # decide log path by stage
                stage = kwargs["stage"]  # should be lookahead or encode
                # stage="encode"
                if stage == "lookahead":
                    granularity = 16
                elif stage == "encode":
                    granularity = 4
                else:
                    raise ValueError("invalid stage")

                # backward motion vector computation
                yuv_file_path = os.path.join(tempdir, f"{idx}_to_{ref_idx}_input.yuv")
                log_base_name = f"{idx}_to_{ref_idx}"
                log_path = os.path.join(log_root, log_base_name + f"-{kwargs.get('preset', 'fast')}-{stage}-{granularity}x{granularity}.csv")
                with open(yuv_file_path, "wb") as f:
                    f.write(cv2.cvtColor(source_frame, cv2.COLOR_BGR2YUV_I420).tobytes())
                    f.write(cv2.cvtColor(target_frame, cv2.COLOR_BGR2YUV_I420).tobytes())

                self.encoder.encode(
                    input_path=yuv_file_path,
                    output_path="/dev/null",
                    log_base_name=log_base_name,
                    log_root=log_root,
                    frame_cnt=2,
                    size=kwargs.get("size", None),
                    frame_rate=kwargs.get("frame_rate", None),
                    lookahead_flag=True,
                    encoding_flag=True,
                    x265_params=x265_params,
                )
                X265MVWrapper._update_flow(log_path, backward_flows, idx, granularity)

                # forward motion vector computation
                yuv_file_path = os.path.join(tempdir, f"{ref_idx}_to_{idx}_input.yuv")
                log_base_name = f"{ref_idx}_to_{idx}"
                log_path = os.path.join(log_root, log_base_name + f"-{kwargs.get('preset', 'fast')}-{stage}-{granularity}x{granularity}.csv")
                with open(yuv_file_path, "wb") as f:
                    f.write(cv2.cvtColor(target_frame, cv2.COLOR_BGR2YUV_I420).tobytes())
                    f.write(cv2.cvtColor(source_frame, cv2.COLOR_BGR2YUV_I420).tobytes())

                self.encoder.encode(
                    input_path=yuv_file_path,
                    output_path="/dev/null",
                    log_base_name=log_base_name,
                    log_root=log_root,
                    frame_cnt=2,
                    size=kwargs.get("size", None),
                    frame_rate=kwargs.get("frame_rate", None),
                    lookahead_flag=True,
                    encoding_flag=True,
                    x265_params=x265_params,
                )
                X265MVWrapper._update_flow(log_path, forward_flows, idx, granularity)

        # Convert to torch tensors
        forward_flows = torch.from_numpy(forward_flows).to(self.device)
        backward_flows = torch.from_numpy(backward_flows).to(self.device)

        return [forward_flows, backward_flows], None
    
    def compute_flow_from_tensors(self, ref_frame_tensor: torch.Tensor, current_frame_tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # print(kwargs)
        if self.native_x265:
            res=self.native_wrapper.compute_flow_from_tensors(ref_frame_tensor,current_frame_tensor, **kwargs)
            return res
            # raise NotImplementedError("compute_flow_from_tensors is not implemented for the native x265 wrapper yet.")

        if ref_frame_tensor.dim() != 4:
            raise ValueError(f"Expected 4D input tensors [B, C, H, W], but got {ref_frame_tensor.dim()}D")
        
        input_device = ref_frame_tensor.device
        B, C, H, W = ref_frame_tensor.shape
        size_str = f"{W}x{H}"
        
        stage = kwargs.get('stage', 'encode')
        # stage="encode"
        if not stage:
            raise ValueError("'stage' ('lookahead' or 'encode') must be provided in kwargs for x265.")
        granularity = 16 if stage == "lookahead" else 4

        forward_flows_np = np.zeros((B, 2, H, W), dtype=np.float32)
        backward_flows_np = np.zeros((B, 2, H, W), dtype=np.float32)

        x265_params = {
            "preset": kwargs.get("preset", "fast"),
            "ctu": kwargs.get("ctu", 16),
            "crf": kwargs.get("crf", 23),
            "enable_p_intra": kwargs.get("enable_p_intra", False),
            "lookahead_intra": kwargs.get("lookahead_intra", False),
        }

        def tensor_to_cv2_img(tensor: torch.Tensor) -> np.ndarray:
            img_np = tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.5 + 0.5) * 255.0
            img_np = img_np.astype(np.uint8)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with tempfile.TemporaryDirectory() as tempdir:
            log_root = os.path.join(tempdir, "x265_logs")
            for i in range(B):
                source_frame_bgr = tensor_to_cv2_img(ref_frame_tensor[i])
                target_frame_bgr = tensor_to_cv2_img(current_frame_tensor[i])

                # --- Backward Flow Computation (current -> ref) ---
                yuv_file_path_bwd = os.path.join(tempdir, f"bwd_{i}_input.yuv")
                log_base_name_bwd = f"bwd_{i}"
                log_path_bwd = os.path.join(log_root, log_base_name_bwd + f"-{kwargs.get('preset', 'fast')}-{stage}-{granularity}x{granularity}.csv")
                with open(yuv_file_path_bwd, "wb") as f:
                    f.write(cv2.cvtColor(source_frame_bgr, cv2.COLOR_BGR2YUV_I420).tobytes())
                    f.write(cv2.cvtColor(target_frame_bgr, cv2.COLOR_BGR2YUV_I420).tobytes())
                
                # --- MODIFICATION: Adopt the 'whitelist' approach ---
                self.encoder.encode(
                    input_path=yuv_file_path_bwd,
                    output_path="/dev/null",
                    log_base_name=log_base_name_bwd,
                    log_root=log_root,
                    frame_cnt=2,
                    size=size_str, # Use the calculated size_str
                    frame_rate=kwargs.get("frame_rate", 30),
                    lookahead_flag=True,
                    encoding_flag=True,
                    x265_params=x265_params,
                    stage=stage
                )
                X265MVWrapper._update_flow(log_path_bwd, backward_flows_np, i, granularity)

                # --- Forward Flow Computation (ref -> current) ---
                yuv_file_path_fwd = os.path.join(tempdir, f"fwd_{i}_input.yuv")
                log_base_name_fwd = f"fwd_{i}"
                log_path_fwd = os.path.join(log_root, log_base_name_fwd + f"-{kwargs.get('preset', 'fast')}-{stage}-{granularity}x{granularity}.csv")
                with open(yuv_file_path_fwd, "wb") as f:
                    f.write(cv2.cvtColor(target_frame_bgr, cv2.COLOR_BGR2YUV_I420).tobytes())
                    f.write(cv2.cvtColor(source_frame_bgr, cv2.COLOR_BGR2YUV_I420).tobytes())

                # --- MODIFICATION: Adopt the 'whitelist' approach ---
                self.encoder.encode(
                    input_path=yuv_file_path_fwd,
                    output_path="/dev/null",
                    log_base_name=log_base_name_fwd,
                    log_root=log_root,
                    frame_cnt=2,
                    size=size_str, # Use the calculated size_str
                    frame_rate=kwargs.get("frame_rate", 30),
                    lookahead_flag=True,
                    encoding_flag=True,
                    x265_params=x265_params,
                    stage=stage
                )
                X265MVWrapper._update_flow(log_path_fwd, forward_flows_np, i, granularity)

        forward_flow_tensor = torch.from_numpy(forward_flows_np).to(input_device)
        backward_flow_tensor = torch.from_numpy(backward_flows_np).to(input_device)

        # print("\n\n\nWarning:x265 params:",kwargs,torch.mean(ref_frame_tensor),torch.mean(current_frame_tensor),torch.mean(forward_flow_tensor),
        #       torch.mean(backward_flow_tensor))
        # if (stage=="lookahead"): return 2*forward_flow_tensor, 2*backward_flow_tensor
        print(ref_frame_tensor.shape,ref_frame_tensor.dtype,current_frame_tensor.shape,current_frame_tensor.dtype ,torch.mean(ref_frame_tensor), torch.mean(current_frame_tensor),kwargs,torch.mean(ref_frame_tensor[:,0,:,:]),torch.mean(ref_frame_tensor[:,1,:,:]),torch.mean(ref_frame_tensor[:,2,:,:]))
        print("Final return of x265",torch.mean(forward_flow_tensor), torch.mean(backward_flow_tensor))
        return forward_flow_tensor, backward_flow_tensor