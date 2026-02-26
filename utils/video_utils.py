import cv2
import numpy as np
import torch


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
    H = int(np.round(H / 32.0)) * 32
    W = int(np.round(W / 32.0)) * 32
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
    
    # Skip to start frame
    while frame_cnt < start_frame_idx:
        success = video_capture.grab()
        if not success:
            raise ValueError("failed to grab frame")
        frame_cnt += 1
    
    # Read frames
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if max_frames is not None and frame_cnt >= max_frames + start_frame_idx:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames += [frame]
        frame_cnt += 1

    video_capture.release()
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