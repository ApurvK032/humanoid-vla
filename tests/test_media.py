from pathlib import Path

import numpy as np
from PIL import Image

from humanoid_vla.utils.media import write_gif


def test_gif_keeps_every_nth_frame_at_the_requested_width(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (60, 120, 3), dtype=np.uint8) for _ in range(10)]
    path = tmp_path / "episode.gif"
    write_gif(frames, path, width=60, every=2, frame_ms=40)
    with Image.open(path) as gif:
        assert gif.size == (60, 30)
        assert gif.n_frames == 5
        assert gif.info["duration"] == 80
