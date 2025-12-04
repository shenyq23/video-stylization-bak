import os
import json

base_param = {
    "crf": 23,
    "preset": "fast",
    "stage": "encode",
}

def run(param):
    param_str = json.dumps(param).replace('"', "'")
    print(f"python main.py --video_name fade --batch_size 4 --flow_model x265 --resolution 512 --x265_params \"{param_str}\"")

run(base_param)