import os
import json

x265_configs = [
    {"crf": 1, "preset": "slow", "stage": "lookahead"}
]

for batch in [1, 2, 3, 4, 6, 50]:
    suffix = "" if batch > 1 else " --discard_key_frames"
    for name in ["music", "boxer", "fade"]:
        os.system(f"python main.py --video_name {name} --batch_size {batch} --flow_model gmflow --resolution 512 --device cuda:1 {suffix}")
        for config in x265_configs:
            config_str = json.dumps(config).replace('"', "'")
            os.system(f"python main.py --video_name {name} --batch_size {batch} --flow_model mix --resolution 512 --x265_params \"{config_str}\" --device cuda:1 {suffix}")
            os.system(f"python main.py --video_name {name} --batch_size {batch} --flow_model reverse_mix --resolution 512 --x265_params \"{config_str}\" --device cuda:1 {suffix}")