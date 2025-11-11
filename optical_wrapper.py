import sys
import gc
import torch

sys.path.append("./deps/gmflow")
sys.path.append("./deps/RAFT")

from gmflow.gmflow import GMFlow
from gmflow.geometry import flow_warp, forward_backward_consistency_check
from deps.RAFT.core.raft import RAFT

class OpticalFlowWrapper:
    def __init__(self, device):
        self.device = device

    def flow_warp(self, frame, flow):
        raise NotImplementedError
    
    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list):
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

    def flow_warp(self, frame, flow):
        return flow_warp(frame, flow)
    
    def compute_flow_and_occlusion(self, frames, ref_frame_idx_list):
        with torch.no_grad():
            images = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames], dim=0).to(self.device)
            results_dict = self.model(
                images,
                images[ref_frame_idx_list],
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
            warped_images = self.flow_warp(images, backward_flows)
            backward_occlusions = torch.clamp(backward_occlusions + (abs(images[ref_frame_idx_list] - warped_images).mean(dim=1) > 255 * 0.25).float(), 0, 1)

            warped_images = self.flow_warp(images[ref_frame_idx_list], forward_flows)
            forward_occlusions = torch.clamp(forward_occlusions + (abs(images - warped_images).mean(dim=1) > 255 * 0.25).float(), 0, 1)

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