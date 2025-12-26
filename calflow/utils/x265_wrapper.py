import os
from .command import Command

class X265EncoderWrapper:
    def __init__(self, encoder_path="/home/holder/video-stylization/bin/x265"):
        self.encoder_path = encoder_path

    def encode(
        self,
        input_path,
        output_path,
        log_base_name,
        log_root,
        frame_cnt,
        preset="fast",
        size=None,
        frame_rate=None,
        lookahead_flag=True,
        encoding_flag=True,
        enable_p_intra=False,
        ctu=16,
        crf=23,
    ):
        assert input_path[-4:] in [".yuv", ".y4m"]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        cmd = (Command(self.encoder_path)
            .add_flag("preset", preset, is_important=True, is_full=True)
            .add_flag("input", input_path, is_important=True, is_full=True)
            .add_flag("output", output_path, is_important=True, is_full=True)
            .add_flag("out-motion-dir", log_root, is_important=True, is_full=True)
            .add_flag("out-motion-name", log_base_name, is_important=True, is_full=True)
            .add_flag("print-motion-info", 2 * int(encoding_flag) + int(lookahead_flag), is_important=True, is_full=True)
            .add_flag("frames", frame_cnt, is_important=True, is_full=True)
            .add_flag("ctu", ctu, is_full=True, is_important=True)
            .add_flag("crf", crf, is_full=True, is_important=True))
        if not enable_p_intra:
            cmd = cmd.add_flag("no-p-intra", is_full=True, is_important=True)
        if input_path[-4:] == ".yuv":
            assert size is not None
            assert frame_rate is not None
            cmd = (cmd
                .add_flag("input-res", size, is_important=True, is_full=True)
                .add_flag("fps", frame_rate, is_important=True, is_full=True))
        cmd.run()
        
if __name__ == "__main__":
    encoder = X265EncoderWrapper()
    encoder.encode(
        input_path="./data/blue_sky_1080p25.y4m",
        output_path="/dev/null",
        log_root="./x265_log/dummy",
        log_base_name="dummy",
        frame_cnt=217,
        lookahead_flag=False,
    )