import os
import shutil
from collections import namedtuple
from command import Command

class X265EncoderWrapper:
    def __init__(self, encoder_path="/home/holder/optical/bin/x265"):
        self.encoder_path = encoder_path

    def encode(
        self,
        input_path,
        output_path,
        log_root,
        frame_cnt,
        size=None,
        frame_rate=None,
        lookahead_flag=True,
        encoding_flag=True,
    ):
        assert input_path[-4:] in [".yuv", ".y4m"]

        if not os.path.exists(log_root):
            os.makedirs(log_root)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        cmd = (Command(self.encoder_path)
            .add_flag("preset", "fast", is_important=True, is_full=True)
            .add_flag("input", input_path, is_important=True, is_full=True)
            .add_flag("output", output_path, is_important=True, is_full=True)
            .add_flag("print-motion-info", 2 * int(encoding_flag) + int(lookahead_flag), is_important=True, is_full=True)
            .add_flag("frames", frame_cnt, is_important=True, is_full=True))
        if input_path[-4:] == ".yuv":
            assert size is not None
            assert frame_rate is not None
            cmd = (cmd
                .add_flag("input-res", size, is_important=True, is_full=True)
                .add_flag("fps", frame_rate, is_important=True, is_full=True))
        cmd.run()

        # move logs
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        for i in range(frame_cnt):
            if lookahead_flag:
                shutil.copy(f"{i}_lowres.txt", log_root)
                os.system(f"rm {i}_lowres.txt")
            if encoding_flag:
                shutil.copy(f"{i}_encoding.txt", log_root)
                os.system(f"rm {i}_encoding.txt")

MVInfo = namedtuple("MVInfo", ["delta_poc", "mvx", "mvy", "weight"])
CUEntry = namedtuple("CUEntry", ["forward_info", "backward_info"])
        
if __name__ == "__main__":
    encoder = X265EncoderWrapper()
    encoder.encode(
        input_path="./data/blue_sky_1080p25.y4m",
        output_path="./data/test.h265",
        log_root="./x265_log/dummy",
        frame_cnt=217,
        lookahead_flag=False,
    )