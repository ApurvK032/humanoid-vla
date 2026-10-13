"""Pinned upstream robot assets.

The Unitree G1 model comes from MuJoCo Menagerie (BSD-3-Clause). Its meshes are about 36 MB, so
they are fetched into ``third_party/`` at a pinned commit (``make assets``) instead of committed.
"""

from pathlib import Path

from humanoid_vla.paths import THIRD_PARTY_DIR

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Latest commit that touched unitree_g1/ when the project started (2026-07-04).
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
MENAGERIE_DIR = THIRD_PARTY_DIR / "mujoco_menagerie"
G1_DIR = MENAGERIE_DIR / "unitree_g1"
FETCH_HINT = "Run `uv run python scripts/fetch_assets.py`."


class AssetError(RuntimeError):
    """Raised when fetched assets are missing or not at the pinned version."""


def menagerie_head(menagerie_dir: Path = MENAGERIE_DIR) -> str | None:
    """Return the checked-out commit of a detached-HEAD Menagerie checkout, or None."""
    head = menagerie_dir / ".git" / "HEAD"
    if not head.is_file():
        return None
    return head.read_text().strip()


def g1_scene_xml(menagerie_dir: Path = MENAGERIE_DIR) -> Path:
    """Path to Menagerie's G1-with-hands scene, after checking it is at the pinned commit."""
    head = menagerie_head(menagerie_dir)
    if head is None:
        raise AssetError(f"Menagerie not found at {menagerie_dir}. {FETCH_HINT}")
    if head != MENAGERIE_COMMIT:
        raise AssetError(
            f"Menagerie is at {head}, expected pinned commit {MENAGERIE_COMMIT}. {FETCH_HINT}"
        )
    return menagerie_dir / "unitree_g1" / "scene_with_hands.xml"
