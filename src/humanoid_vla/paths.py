"""Repository paths.

The package is used from a source checkout (``uv sync`` installs it in editable mode).
Set ``HUMANOID_VLA_ROOT`` to point at a different checkout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HUMANOID_VLA_ROOT", Path(__file__).resolve().parents[2]))

# Upstream assets fetched at pinned versions (gitignored, see THIRD_PARTY.md).
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
# Generated artifacts: images, videos, logs, datasets, checkpoints (gitignored).
OUTPUTS_DIR = REPO_ROOT / "outputs"
# Small measurement files that are committed, each with its provenance.
RESULTS_DIR = REPO_ROOT / "results"
