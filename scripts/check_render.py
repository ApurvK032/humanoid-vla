"""Milestone 0 check: headless rendering and simulation throughput.

Builds the tabletop scene, renders every camera, and measures physics and rendering rates.
Images go to outputs/check_render/. With --save, measurements go to results/m0/render_check.json.

    uv run python scripts/check_render.py [--save]
"""

import argparse
import os
import sys
import time

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()

import mujoco
from PIL import Image

from humanoid_vla.paths import OUTPUTS_DIR, RESULTS_DIR
from humanoid_vla.sim.render import FAST_QUALITY, FULL_QUALITY, CameraRenderer, gl_renderer_name
from humanoid_vla.sim.scene import build_model
from humanoid_vla.utils.provenance import write_result

GL_ENV_VARS = ("MUJOCO_GL", "GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME")
POLICY_SIZE = (256, 256)  # (height, width) candidate policy-camera resolution
VIDEO_SIZE = (480, 640)


def physics_rate(model: mujoco.MjModel, seconds: float) -> dict[str, float]:
    data = mujoco.MjData(model)
    steps = round(seconds / model.opt.timestep)
    start = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start
    return {
        "timestep_s": model.opt.timestep,
        "steps_per_s": steps / elapsed,
        "realtime_factor": steps * model.opt.timestep / elapsed,
    }


def render_rate(renderer: CameraRenderer, data: mujoco.MjData, camera: str, frames: int) -> float:
    renderer.render(data, camera)  # warm-up
    start = time.perf_counter()
    for _ in range(frames):
        renderer.render(data, camera)
    return frames / (time.perf_counter() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=100, help="frames per rendering measurement")
    parser.add_argument("--save", action="store_true", help="write results/m0/render_check.json")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cameras = [model.camera(i).name for i in range(model.ncam)]
    image_dir = OUTPUTS_DIR / "check_render"
    image_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "gl_env": {name: os.environ.get(name) for name in GL_ENV_VARS},
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "nbody": model.nbody},
        "physics": physics_rate(model, seconds=10.0),
        "render_fps": {},
    }

    for size, label in ((POLICY_SIZE, "policy"), (VIDEO_SIZE, "video")):
        for quality, quality_name in ((FULL_QUALITY, "full"), (FAST_QUALITY, "fast")):
            with CameraRenderer(model, *size, quality=quality) as renderer:
                result.setdefault("gl_renderer", gl_renderer_name())
                for camera in cameras:
                    image = renderer.render(data, camera)
                    if image.std() < 1.0:
                        print(f"error: {camera} rendered a blank image", file=sys.stderr)
                        return 1
                    Image.fromarray(image).save(image_dir / f"{camera}_{label}_{quality_name}.png")
                    fps = render_rate(renderer, data, camera, args.frames)
                    key = f"{camera}_{size[1]}x{size[0]}_{quality_name}"
                    result["render_fps"][key] = round(fps, 1)

    print(f"GL renderer: {result['gl_renderer']}")
    print(
        f"Physics: {result['physics']['steps_per_s']:.0f} steps/s "
        f"({result['physics']['realtime_factor']:.1f}x real time)"
    )
    for key, fps in result["render_fps"].items():
        print(f"  {key:28s} {fps:6.1f} fps")
    print(f"Images: {image_dir}")
    if args.save:
        path = RESULTS_DIR / "m0" / "render_check.json"
        write_result(path, result)
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
