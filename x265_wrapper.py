import os
from command import Command

class X265EncoderWrapper:
    def __init__(self, encoder_path):
        self.encoder_path = encoder_path

    def encode(self, input_path, output_path, lookahead_flag=True, encoding_flag=True):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        (Command(self.encoder_path)
            .add_flag("input", input_path, is_important=True)
            .add_flag("output", output_path, is_important=True)
            .add_flag("print-motion-info", 2 * int(encoding_flag) + int(lookahead_flag), is_important=True)
            .add_flag("preset", "medium", is_important=True)
            .run())
        
if __name__ == "__main__":
    encoder = X265EncoderWrapper("./bin/x265")
    encoder.encode(
        input_path="./data/blue_sky_1080p25.y4m",
        output_path="./data/test.mp4",
    )