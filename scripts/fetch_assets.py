"""Fetch MuJoCo Menagerie's unitree_g1/ at the pinned commit into third_party/.

Uses a shallow, sparse, blob-filtered fetch, so only the G1 files (~36 MB) are downloaded.
Safe to re-run: an existing checkout is moved to the pinned commit.

    uv run python scripts/fetch_assets.py
"""

import subprocess
import sys

from humanoid_vla.sim.assets import (
    MENAGERIE_COMMIT,
    MENAGERIE_DIR,
    MENAGERIE_URL,
    g1_scene_xml,
    menagerie_head,
)


def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(MENAGERIE_DIR), *args], check=True)


def main() -> int:
    if menagerie_head() == MENAGERIE_COMMIT:
        print(f"Menagerie already at {MENAGERIE_COMMIT[:12]}: {MENAGERIE_DIR}")
        return 0
    if not (MENAGERIE_DIR / ".git").is_dir():
        MENAGERIE_DIR.mkdir(parents=True, exist_ok=True)
        git("init", "--quiet")
        git("remote", "add", "origin", MENAGERIE_URL)
        git("sparse-checkout", "set", "unitree_g1")
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT)
    git("checkout", "--quiet", "--detach", MENAGERIE_COMMIT)
    print(f"Fetched Menagerie {MENAGERIE_COMMIT[:12]}: {g1_scene_xml()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
