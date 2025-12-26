from main import run_motion_compensation

x265_configs = [
    {"crf": 1, "preset": "slow", "stage": "lookahead"},
    {"crf": 1, "preset": "slow", "stage": "encode"}
]

for batch in [4, 16]:
    for name in [
        "fade",
        "basketball",
        "basketball-drill",
        "basketball-pass",
        "blowing-bubbles",
        "bq-mall",
        "bq-square",
        "bq-terrace",
        "bunny-1",
        "bunny-2",
        "bunny-3",
        "cactus",
        "four-people",
        "johnny",
        "kristen-and-sara",
        "music",
        "party",
        "race-horse",
        "race-horse-lowreq",
    ]:
        run_motion_compensation(video_name=name, batch_size=batch, flow_model="gmflow", resolution=512, device="cuda:1", discard_key_frames=(batch == 1), use_luminosity=True, use_geometry=True)
        for config in x265_configs:
            config_str = " ".join([f"{k}={v}" for k, v in config.items()])
            run_motion_compensation(video_name=name, batch_size=batch, flow_model="x265", resolution=512, x265_params=config_str, device="cuda:1", discard_key_frames=(batch == 1), use_structure=True)
            run_motion_compensation(video_name=name, batch_size=batch, flow_model="x265", resolution=512, x265_params=config_str, device="cuda:1", discard_key_frames=(batch == 1), use_color=True)