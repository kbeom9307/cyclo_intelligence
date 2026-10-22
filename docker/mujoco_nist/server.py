"""Run the standalone headless MuJoCo bridge; rendering never advances physics."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import base64
from io import BytesIO
import json
import signal
import time
import zlib

import mujoco
import numpy as np
from PIL import Image

from adapter import MujocoNistAdapter
from transport import Bridge, FRAME_PROFILES


class Cameras:
    def __init__(self, adapter):
        self.adapter = adapter
        self.robot = self.overview = None
        self.sequence = 0
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0

    def render(self, name):
        overview = name == "spectator"
        if overview and self.overview is None:
            self.overview = mujoco.Renderer(self.adapter.model, height=1440, width=1920)
        if not overview and self.robot is None:
            self.robot = mujoco.Renderer(self.adapter.model, height=360, width=640)
        renderer = self.overview if overview else self.robot
        renderer.update_scene(self.adapter.data, camera=name, scene_option=self.option)
        return renderer.render().copy()

    def capture(self):
        size = self.adapter.c.get("image_size", 128) if self.adapter.c.get("learning_backend") == "hil_serl" else 64
        if size not in (64, 128, 256):
            raise ValueError("Unsupported training image size")
        crops = self.adapter.c.get("camera_crops", {}) if self.adapter.c.get("learning_backend") == "hil_serl" else {}
        names = ["head", "left_wrist", "right_wrist"]
        images = []
        for name in names:
            rgb = self.render(name)
            left, top, right, bottom = crops.get(name, [0, 0, 1, 1])
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("Invalid normalized camera crop")
            height, width = rgb.shape[:2]
            rgb = rgb[int(top*height):max(int(top*height)+1, int(bottom*height)),
                      int(left*width):max(int(left*width)+1, int(right*width))]
            images.append(np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)).transpose(2, 0, 1))
        self.sequence += 1
        pixels = np.concatenate(images).astype(np.uint8)
        return dict(encoding="zlib-uint8", shape=[9, size, size], cameras=names, crops=crops,
                    sequence=self.sequence, timestamp=time.time(),
                    data=base64.b64encode(zlib.compress(pixels.tobytes())).decode("ascii"))

    def close_overview(self):
        if self.overview is not None:
            self.overview.close()
            self.overview = None

    def close(self):
        self.close_overview()
        if self.robot is not None:
            self.robot.close()
            self.robot = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7126)
    parser.add_argument("--round-peg-clearance-mm", type=float, default=.5)
    args = parser.parse_args()
    adapter = MujocoNistAdapter(args.round_peg_clearance_mm)
    cameras = Cameras(adapter)
    adapter.capture_observation = cameras.capture
    bridge = Bridge(adapter, port=args.port)
    bridge.details["visualization"] = dict(enabled=True, width=1920, height=1440, max_fps=4,
                                          views=["spectator", "head", "left_wrist", "right_wrist"],
                                          quality_profiles=list(FRAME_PROFILES), overview_rendering=False)
    running = True

    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    bridge.start()
    print(json.dumps({"ready": True, "port": args.port, **adapter.describe()}), flush=True)
    next_frame = 0.
    try:
        while running:
            bridge.pump(timeout=.01)
            requested = [name for name in ("spectator", "head", "left_wrist", "right_wrist") if bridge.wants_frame(name)]
            if "spectator" not in requested and cameras.overview is not None:
                cameras.close_overview()
                bridge.clear_frames("spectator")
            if requested and time.monotonic() >= next_frame:
                for name in requested:
                    native = Image.fromarray(cameras.render(name))
                    for quality, profile in FRAME_PROFILES.items():
                        width, height = profile[:2] if name == "spectator" else profile[2:4]
                        stream = BytesIO()
                        native.resize((width, height), Image.Resampling.BILINEAR).save(stream, "JPEG", quality=profile[4])
                        bridge.publish_frame(stream.getvalue(), camera=name, quality=quality)
                next_frame = time.monotonic()+.25
            bridge.details["visualization"]["overview_rendering"] = cameras.overview is not None
    finally:
        bridge.close()
        cameras.close()


if __name__ == "__main__":
    main()
