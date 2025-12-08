import cv2

def extract_one_frame_from_video(video_path, frame_idx, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

import os

for file in os.listdir("./output/music"):
    in_path = os.path.join("./output/music", file)
    if not file.endswith(".mp4"):
        continue

    out_path = os.path.join("./output/music_frames", file[:-4] + ".png")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    extract_one_frame_from_video(in_path, 27, out_path)

exit()

import os
import json

base_param = {
    "crf": 23,
    "preset": "fast",
    "stage": "encode",
}

def run(param):
    param_str = json.dumps(param).replace('"', "'")
    os.system(f"python main.py --video_name music --batch_size 4 --flow_model mix --resolution 512 --x265_params \"{param_str}\"")

run(base_param)

base_param["crf"] = 8
run(base_param)
base_param["crf"] = 23

base_param["preset"] = "slow"
run(base_param)
base_param["preset"] = "fast"

base_param["stage"] = "lookahead"
run(base_param)
base_param["stage"] = "encode"

exit()
