"""Tabletop sorting scene: a fixed-base G1, a table, colored blocks and two bins.

Geometry here is a first pass for Milestone 0. Milestone 1 tunes the table, bins and block region
to the arm's measured reachable workspace and adds seeded resets.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_vla.sim.assets import g1_scene_xml
from humanoid_vla.sim.g1 import G1Config, make_fixed_base

RGBA = tuple[float, float, float, float]

BLOCK_COLORS: dict[str, RGBA] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.65, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
}


@dataclass(frozen=True)
class SceneConfig:
    robot: G1Config = field(default_factory=G1Config)

    table_center_xy: tuple[float, float] = (0.34, -0.10)
    table_half_size_xy: tuple[float, float] = (0.20, 0.38)
    table_height: float = 0.78  # top surface z
    table_thickness: float = 0.03

    # 5 cm cubes: wide enough that both fingers of the claw grasp land on the front face.
    block_half_size: float = 0.025
    block_mass: float = 0.05
    # Default block positions (x, y); environments place blocks at reset.
    block_xy: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "red": (0.22, -0.10),
            "green": (0.22, -0.20),
            "blue": (0.22, -0.30),
        }
    )

    # Bins behind the block strip. Both sit inside the region the right arm reaches palm-down
    # without its upper arm touching the torso (docs/DECISIONS.md, D-014).
    bin_xy: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"left": (0.40, -0.06), "right": (0.40, -0.30)}
    )
    bin_inner_half_size: float = 0.055
    bin_wall_height: float = 0.05
    bin_wall_thickness: float = 0.006

    # Head camera: front of the head, pitched down and turned toward the right arm's workspace
    # so both bins are in view. 66 degrees is close to the RealSense D435's horizontal FOV.
    head_camera_pos: tuple[float, float, float] = (0.085, 0.0, 0.386)  # in torso_link frame
    head_camera_pitch_deg: float = 62.0
    head_camera_yaw_deg: float = -20.0  # negative turns toward the robot's right
    head_camera_fovy_deg: float = 66.0
    # External camera for videos and figures (not a policy input).
    scene_camera_pos: tuple[float, float, float] = (1.0, -0.95, 1.30)
    scene_camera_target: tuple[float, float, float] = (0.25, -0.15, 0.90)
    scene_camera_fovy_deg: float = 42.0


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec.from_file(str(g1_scene_xml()))
    spec.modelname = "g1_tabletop_sort"
    # Offscreen framebuffer bounds the largest image any camera can render.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    # Contact settings MuJoCo recommends for grasping: elliptic friction cones and a high
    # impedance ratio, which make frictional contacts much less prone to slip.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    make_fixed_base(spec, cfg.robot)
    _add_table(spec, cfg)
    _add_bins(spec, cfg)
    _add_blocks(spec, cfg)
    _add_cameras(spec, cfg)
    return spec


def build_model(cfg: SceneConfig | None = None) -> mujoco.MjModel:
    return build_spec(cfg).compile()


def _add_table(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    hx, hy = cfg.table_half_size_xy
    top_z = cfg.table_height - cfg.table_thickness / 2
    table = spec.worldbody.add_body(name="table", pos=[*cfg.table_center_xy, 0.0])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, cfg.table_thickness / 2],
        pos=[0, 0, top_z],
        rgba=[0.62, 0.47, 0.33, 1],
        friction=[1.0, 0.005, 0.0001],
    )
    leg_half_height = (cfg.table_height - cfg.table_thickness) / 2
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        table.add_geom(
            name=f"table_leg_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, leg_half_height],
            pos=[sx * (hx - 0.03), sy * (hy - 0.03), leg_half_height],
            rgba=[0.45, 0.33, 0.23, 1],
            contype=0,
            conaffinity=0,
        )


def _add_bins(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    inner, wall, height = cfg.bin_inner_half_size, cfg.bin_wall_thickness, cfg.bin_wall_height
    outer = inner + wall
    rgba = [0.88, 0.88, 0.85, 1]
    for name, xy in cfg.bin_xy.items():
        body = spec.worldbody.add_body(name=f"bin_{name}", pos=[*xy, cfg.table_height])
        body.add_geom(
            name=f"bin_{name}_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[outer, outer, wall / 2],
            pos=[0, 0, wall / 2],
            rgba=rgba,
        )
        walls = {
            "px": ([inner + wall / 2, 0], [wall / 2, outer]),
            "nx": ([-(inner + wall / 2), 0], [wall / 2, outer]),
            "py": ([0, inner + wall / 2], [inner, wall / 2]),
            "ny": ([0, -(inner + wall / 2)], [inner, wall / 2]),
        }
        for wall_name, (pos_xy, half_xy) in walls.items():
            body.add_geom(
                name=f"bin_{name}_wall_{wall_name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[*half_xy, height / 2],
                pos=[*pos_xy, wall + height / 2],
                rgba=rgba,
            )


def _add_blocks(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    h = cfg.block_half_size
    for color, xy in cfg.block_xy.items():
        body = spec.worldbody.add_body(name=f"block_{color}", pos=[*xy, cfg.table_height + h])
        body.add_freejoint(name=f"block_{color}_joint")
        body.add_geom(
            name=f"block_{color}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[h, h, h],
            mass=cfg.block_mass,
            rgba=list(BLOCK_COLORS[color]),
            friction=[1.0, 0.005, 0.0001],
        )


def _add_cameras(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    pitch = np.deg2rad(cfg.head_camera_pitch_deg)
    yaw = np.deg2rad(cfg.head_camera_yaw_deg)
    head = spec.body("torso_link").add_camera(
        name="head", pos=list(cfg.head_camera_pos), fovy=cfg.head_camera_fovy_deg
    )
    # The torso frame is axis-aligned with the world while the waist joints are frozen at zero.
    view = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    pos = np.asarray(cfg.head_camera_pos)
    _set_xyaxes(head, look_at_xyaxes(tuple(pos), tuple(pos + view)))

    scene = spec.worldbody.add_camera(
        name="scene", pos=list(cfg.scene_camera_pos), fovy=cfg.scene_camera_fovy_deg
    )
    _set_xyaxes(scene, look_at_xyaxes(cfg.scene_camera_pos, cfg.scene_camera_target))


def _set_xyaxes(camera: mujoco.MjsCamera, xyaxes: list[float]) -> None:
    camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    camera.alt.xyaxes = xyaxes


def look_at_xyaxes(
    pos: tuple[float, float, float], target: tuple[float, float, float]
) -> list[float]:
    """MuJoCo camera ``xyaxes`` for a camera at ``pos`` looking at ``target`` with +z up."""
    back = np.asarray(pos, dtype=float) - np.asarray(target, dtype=float)  # cameras look along -z
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return [*right, *up]
