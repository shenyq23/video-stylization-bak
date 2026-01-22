import os
from utils.command import Command

class X265EncoderWrapper:
    DEFAULT_PARAMS = {
        "preset": "fast",
        "ctu": 16,
        "crf": 23,
        "enable_p_intra": False,
    }

    def __init__(self, encoder_path=None):
        if encoder_path is None:
            self.encoder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "x265")
        else:
            self.encoder_path = encoder_path

    def encode(
        self,
        input_path,
        output_path,
        log_base_name,
        log_root,
        frame_cnt,
        size=None,
        frame_rate=None,
        lookahead_flag=True,
        encoding_flag=True,
        x265_params=None,
    ):
        assert input_path[-4:] in [".yuv", ".y4m"]

        # merge default params with user-provided params
        params = self.DEFAULT_PARAMS.copy()
        if x265_params:
            params.update(x265_params)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        cmd = (Command(self.encoder_path)
            .add_flag("preset", params["preset"], is_important=True, is_full=True)
            .add_flag("input", input_path, is_important=True, is_full=True)
            .add_flag("output", output_path, is_important=True, is_full=True)
            .add_flag("out-motion-dir", log_root, is_important=True, is_full=True)
            .add_flag("out-motion-name", log_base_name, is_important=True, is_full=True)
            .add_flag("print-motion-info", 2 * int(encoding_flag) + int(lookahead_flag), is_important=True, is_full=True)
            .add_flag("frames", frame_cnt, is_important=True, is_full=True)
            .add_flag("ctu", params["ctu"], is_full=True, is_important=True)
            .add_flag("crf", params["crf"], is_full=True, is_important=True))

        if not params["enable_p_intra"]:
            cmd = cmd.add_flag("no-p-intra", is_full=True, is_important=True)

        handled_keys = {"preset", "ctu", "crf", "enable_p_intra"}
        for key, value in params.items():
            if key not in handled_keys:
                cli_key = key.replace("_", "-")
                if value is None or value is True:
                    # Boolean flag (e.g., --no-wpp)
                    cmd = cmd.add_flag(cli_key, is_full=True, is_important=False)
                elif value is not False:
                    # Key-value parameter
                    cmd = cmd.add_flag(cli_key, value, is_full=True, is_important=False)

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