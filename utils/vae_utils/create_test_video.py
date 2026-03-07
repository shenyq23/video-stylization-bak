import os
import cv2
import numpy as np

def make_step_translation_video(
    img_path: str,
    T: int = 13,
    dx: int = 10,
    dy: int = 0,
    block_size: int = 4,          # “1,4,4,4...”里的 4
    first_move_at: int = 1,       # 0-based：第2帧发生第一次移动 -> t=1
    fill_bgr=(0, 255, 0),         # （BGR）
    out_dir: str = "./out",
    out_mp4: str = "out.mp4",
    fps: int = 10,
):
    """
    生成“阶梯式平移”视频：
    - frame0: 原图
    - 从 frame1 开始：每隔 block_size 帧移动一次(dx,dy)，其余帧保持不变
      例如 T=9, block_size=4, first_move_at=1:
      t=0 原图
      t=1 移动
      t=2,3,4 静止
      t=5 移动
      t=6,7,8 静止

    返回：
    - video: (T,H,W,3) uint8
    - flow : (T-1,H,W,2) float32  相邻帧光流
    - mask : (T-1,H,W)   uint8    1=新露出无效区域, 0=有效
    """
    os.makedirs(out_dir, exist_ok=True)

    bgr0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr0 is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    H, W = bgr0.shape[:2]
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    video[0] = bgr0

    # 预先准备一次平移的矩阵
    # x' = x + dx
    # y' = y + dy
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)


    # fill_colors = [
    #     (0,   0, 255),   # 红
    #     (0, 255,   0),   # 绿
    #     (255, 0,   0),   # 蓝
    #     (0, 255, 255),   # 黄
    #     (255, 0, 255),   # 紫
    #     (255, 255, 0),   # 青
    # ]
    fill_colors = [
        (0,   0, 255),   # 红
        (0, 255,   0),   # 绿
        (255, 0,   0),   # 蓝
    ]

    def do_move_step(img): return cv2.warpAffine( img, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_bgr, )

    move_counter = 0
    # 生成视频帧：动/静
    for t in range(1, T):
        # t=1, 1+block_size, 1+2*block_size, ... 发生移动
        if t >= first_move_at and ((t - first_move_at) % block_size == 0):
           video[t] = do_move_step(video[t - 1])
           fill_bgr = (255, 0, 0)
            # color = fill_colors[move_counter % len(fill_colors)]
            # video[t] = cv2.warpAffine(
                # video[t - 1],
                # M,
                # (W, H),
                # flags=cv2.INTER_NEAREST,
                # borderMode=cv2.BORDER_CONSTANT,
                # borderValue=color,
            # )
            # move_counter += 1
        else:
            video[t] = video[t - 1]  # 静止：直接拷贝上一帧

    # ========= paths =========
    mp4_path = os.path.join(out_dir, out_mp4)
    frames_dir = os.path.join(out_dir, out_mp4.replace(".mp4", "_frames"))
    os.makedirs(frames_dir, exist_ok=True)

    # ========= video writer =========
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {mp4_path}")

    # ========= write =========
    for t in range(T):
        frame = video[t]  # (H, W, 3), uint8, BGR

        # 写视频
        writer.write(frame)

        # 同时保存单帧
        cv2.imwrite(
            os.path.join(frames_dir, f"frame_{t:05d}.png"),
            frame
        )

    writer.release()
    print(f"[OK] video saved to: {mp4_path}")
    print(f"[OK] frames saved to: {frames_dir}")




    # avi_path = os.path.join(out_dir, "input.avi")  # 改后缀
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # writer = cv2.VideoWriter(avi_path, fourcc, fps, (W, H))
    # if not writer.isOpened():
    #     raise RuntimeError(f"Failed to open VideoWriter: {avi_path}")

    # for t in range(T):
    #     writer.write(video[t])
    # writer.release()

    # size = os.path.getsize(avi_path)
    # print(f"[OK] video saved to: {avi_path}, size={size} bytes")

    cap = cv2.VideoCapture(mp4_path)
    ok, frame = cap.read()
    print("readback ok =", ok, "frame shape =", None if not ok else frame.shape)
    print("frame count =", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()


    # flow: (H, W, 2), 全是 (dx, dy)
    # 注意是: backward flow
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = -dx
    flow[..., 1] = -dy
    flow += np.random.normal(0, 1, size=(H, W, 2)).astype(np.float32)

    print(f"flow mean: {flow.mean():.4f}")
    # mask: (H, W), 左 dx 列为 1
    mask = np.zeros((H, W), dtype=np.uint8)

    if dy > 0:
        mask[:dy, :] = 1
    elif dy < 0:
        mask[H + dy:, :] = 1

    if dx > 0:
        mask[:, :dx] = 1
    elif dx < 0:
        mask[:, W + dx:] = 1

    # 保存
    flow_path = os.path.join(out_dir, "flow.npy")
    mask_path = os.path.join(out_dir, "mask.npy")
    np.save(flow_path, flow)
    np.save(mask_path, mask)

    print(f"[OK] flow saved to {flow_path}, shape={flow.shape}")
    print(f"[OK] mask saved to {mask_path}, shape={mask.shape}")



    return video, flow, mask


if __name__ == "__main__":
    video, flow, mask = make_step_translation_video(
        img_path="assets/bird_first_frame.png",
        T=125,
        dx=20,
        dy=0,           # 只向右移动；如果也要向下就改 dy=10
        block_size=4,   # “1,4,4,4...”
        first_move_at=5,
        fill_bgr=(0, 0, 255),
        out_dir="assets",
        out_mp4="input_create_bird.mp4",
        fps=10,
    )
