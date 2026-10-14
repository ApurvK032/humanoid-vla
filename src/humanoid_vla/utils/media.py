"""Small looping GIFs of recorded episodes for the documentation."""

from pathlib import Path

import numpy as np
from PIL import Image


def write_gif(
    frames: list[np.ndarray], path: Path, width: int = 480, every: int = 2, frame_ms: int = 40
) -> None:
    """Write every ``every``-th frame, resized to ``width`` pixels, as a looping GIF.

    ``frame_ms`` is the time between consecutive input frames (40 ms at the 25 Hz control rate).
    """
    height = round(frames[0].shape[0] * width / frames[0].shape[1])
    images = [
        Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS)
        for f in frames[::every]
    ]
    # One palette for all frames, built from frames across the episode so every block color
    # keeps its hue.
    samples = images[:: max(1, len(images) // 8)]
    montage = Image.new("RGB", (images[0].width, images[0].height * len(samples)))
    for k, sample in enumerate(samples):
        montage.paste(sample, (0, k * images[0].height))
    palette = montage.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms * every, loop=0)
