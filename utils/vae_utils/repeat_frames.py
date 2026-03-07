import cv2
import os

def repeat_frames(input_path: str, output_path: str, repeat_times: int = 4):
    """
    repeat_times=4 表示每帧输出4次（原帧+重复3次）
    前 50 帧额外保存为图片, 进行检查
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[info] fps={fps}, size=({w},{h}), frames={total}, repeat_times={repeat_times}")
    # 输出保持原fps => 时长变为 repeat_times 倍
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 也可用 'avc1' 看你环境
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # === 新增：图片保存目录 ===
    save_img_dir = "../assets/first_50_frames"
    os.makedirs(save_img_dir, exist_ok=True)

    idx = 0
    out_idx = 0      # 最终视频帧索引（关键）
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        repeat = repeat_times + 1 if idx == 0 else repeat_times

        for _ in range(repeat):
            # 写入视频
            out.write(frame)

            # === 如果是最终视频前 50 帧，保存图片 ===
            if out_idx < 50:
                img_path = os.path.join(
                    save_img_dir, f"frame_{out_idx:04d}.png"
                )
                cv2.imwrite(img_path, frame)

            out_idx += 1

        idx += 1
        if idx % 100 == 0:
            print(f"[progress] read {idx}/{total}, written {out_idx} frames")


    cap.release()
    out.release()
    print(f"[done] saved to: {output_path}")

if __name__ == "__main__":
    repeat_frames("../assets/cat.mp4", "../assets/cat_repeat.mp4", repeat_times=4)
