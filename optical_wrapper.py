import os
import sys
import gc
import torch
import tempfile
import cv2

sys.path.append("./deps/gmflow")
sys.path.append("./deps/RAFT")

from x265_wrapper import X265EncoderWrapper, MVInfo, CUEntry
from gmflow.gmflow import GMFlow
from gmflow.geometry import flow_warp, forward_backward_consistency_check
from deps.RAFT.core.raft import RAFT

# the warp function from gmflow can be used universally
def universal_flow_warp(frame, flow):
    return flow_warp(frame, flow)

class OpticalFlowWrapper:
    def __init__(self, device):
        self.device = device
    
    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list, **kwargs):
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
            
        checkpoint = torch.load("./deps/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth", map_location=lambda storage, loc: storage)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        optical_flow_model.load_state_dict(weights, strict=False)
        optical_flow_model.eval()

        for param in optical_flow_model.parameters():
            param.requires_grad = False

        return optical_flow_model
    
    def __init__(self, device):
        super().__init__(device)
        self.model = self._load_gmflow_model()
    
    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list, **kwargs):
        """ no other kwargs needed for gmflow
        """
        with torch.no_grad():
            images = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames], dim=0).to(self.device)
            results_dict = self.model(
                images[ref_frame_idx_list],
                images,
                attn_splits_list=[2], 
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=True,
            )

            flow_prediction = results_dict["flow_preds"][-1]  # [2*B, 2, H, W]
            forward_flows, backward_flows = flow_prediction.chunk(2)  # [B, 2, H, W]
            forward_occlusions, backward_occlusions = forward_backward_consistency_check(forward_flows, backward_flows)  # [B, H, W]
            
            # params of flow_warp
            # 1. source image
            # 2. the flow from target to source
            warped_images = flow_warp(images[ref_frame_idx_list], backward_flows)
            backward_occlusions = torch.clamp(backward_occlusions + (abs(images - warped_images).mean(dim=1) > 255 * 0.25).float(), 0, 1)

            warped_images = flow_warp(images, forward_flows)
            forward_occlusions = torch.clamp(forward_occlusions + (abs(images[ref_frame_idx_list] - warped_images).mean(dim=1) > 255 * 0.25).float(), 0, 1)

            gc.collect()
            torch.cuda.empty_cache()
            
            return [forward_flows, backward_flows], [forward_occlusions, backward_occlusions]

class RAFTFlowWrapper(OpticalFlowWrapper):
    def __init__(self, device):
        super().__init__(device)
        self.model = torch.nn.DataParallel(RAFT())
        self.model.load_state_dict(torch.load("./deps/RAFT/models/raft-things.pth"))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()
    
    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list, **kwargs):
        raise NotImplementedError

class X265MVWrapper(OpticalFlowWrapper):
    def __init__(self, device, encoder_path="/home/holder/optical/bin/x265"):
        super().__init__(device)
        self.encoder = X265EncoderWrapper(encoder_path)

    @staticmethod
    def _parse_section(s):
        s = s.strip()
        assert s[0] == "(" and s[-1] == ")"
        s = [x.strip() for x in s[1:-1].split(",")]
        assert len(s) == 4
        return MVInfo(
            delta_poc=int(s[0]),
            mvx=int(s[1]),
            mvy=int(s[2]),
            weight=int(s[3]),
        )

    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list, **kwargs):
        """ kwargs should contain size and frame rate
        """
        with tempfile.TemporaryDirectory() as tempdir:
            print(f"encoding in temporary path: {tempdir}")
            yuv_file_path = os.path.join(tempdir, "input.yuv")
            output_file_path = os.path.join(tempdir, "output.h265")
            log_root = os.path.join(tempdir, "encoding_log")
            with open(yuv_file_path, "wb") as f:
                for frame in frames:
                    f.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
            self.encoder.encode(
                input_path=yuv_file_path,
                output_path=output_file_path,
                log_root=log_root,
                frame_cnt=len(frames),
                size=kwargs["size"],
                frame_rate=kwargs["frame_rate"],
            )

            width = int(kwargs["size"].split("x")[0])
            height = int(kwargs["size"].split("x")[1])
            frame_mv_infos = []
            for i in range(len(frames)):
                with open(os.path.join(log_root, f"{i}_encoding.txt"), "r") as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines]
                    lines = [line for line in lines if line != ""]
                    assert len(lines) == width * height / 16

                    mv_infos = []
                    for line in lines:
                        section_cnt = line.count("(")
                        assert section_cnt in [1, 2]
                        if section_cnt == 2:
                            sections = [s for s in line.split("),(")]
                            assert len(sections) == 2
                            mv_infos.append(CUEntry(
                                forward_info=X265MVWrapper._parse_section(sections[0] + ")"),
                                backward_info=X265MVWrapper._parse_section("(" + sections[1]),
                            ))
                        else:
                            mv_infos.append(CUEntry(
                                forward_info=X265MVWrapper._parse_section(line),
                                backward_info=None,
                            ))
                    frame_mv_infos.append(mv_infos)

            # further processing, ready for flow_warp