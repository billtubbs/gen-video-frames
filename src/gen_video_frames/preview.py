"""
Previewing LED frame data as it will look on the display: each LED drawn
as a dot at its real position, so a sequence can be checked before
copying it to the Pi.

Colours are the output values with the gamma/dimness scaling undone
(scale_led_values()), i.e. the source colours as each LED shows them, at
full brightness.
"""

from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gen_video_frames.video import write_video

DOT_SIZE = 0.8  # dot diameter, as a fraction of the typical LED spacing
MARGIN = 10  # pixels around the display in each image


def load_led_centres():
    """(x, y) centre of each LED, shape (1593, 2), y = 0 at the top."""
    path = resources.files("gen_video_frames") / "data/led_centres_1593.csv"
    with path.open() as f:
        return np.loadtxt(f, delimiter=",", comments="#", skiprows=4)


def led_index_image(centres, size):
    """
    A size x size array giving the LED drawn at each pixel (-1 for none),
    so a frame of LED colours renders as colours[index_image].
    """
    xy = centres - centres.min(axis=0)
    scale = (size - 2 * MARGIN) / xy.max()
    xy = MARGIN + xy * scale
    # Typical spacing: median distance from each LED to its nearest one
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(d, np.inf)
    radius = DOT_SIZE * np.median(d.min(axis=1)) / 2

    index = np.full((size, size), -1, dtype=int)
    rows, cols = np.mgrid[0:size, 0:size]
    for i, (x, y) in enumerate(xy):
        r0, r1 = int(y - radius), int(y + radius) + 2
        c0, c1 = int(x - radius), int(x + radius) + 2
        inside = (rows[r0:r1, c0:c1] - y) ** 2 + (
            cols[r0:r1, c0:c1] - x
        ) ** 2 <= radius**2
        index[r0:r1, c0:c1][inside] = i
    return index


def unscale_led_values(z, dimness=6):
    """Inverse of led_frames.scale_led_values(), back to 0-255 colours."""
    raw = np.sqrt(z.astype(float) * 256 * dimness)
    return np.clip(np.round(raw), 0, 255).astype("uint8")


def render_frames(colours, index_image):
    """
    Renders LED colours, shape (n_frames, 1593, 3), to RGB images, shape
    (n_frames, size, size, 3).
    """
    padded = np.concatenate(
        [colours, np.zeros((len(colours), 1, 3), dtype="uint8")], axis=1
    )
    # index -1 picks the black padding LED
    return padded[:, index_image]


def preview(params):
    """
    Renders OUTPUT_FILE as it will look on the display:

    - PREVIEW_IMAGE (default: OUTPUT_FILE name + "_preview.png"): the
      frames listed in PREVIEW_FRAMES (indices into the output; default 4
      evenly spaced) side by side, each PREVIEW_SIZE pixels square
      (default 400).
    - PREVIEW_VIDEO (optional): every frame, at PREVIEW_FPS (default 24).
    """
    output_path = Path(params["OUTPUT_FILE"])
    led_frames = np.load(output_path)["led_frames"]
    colours = unscale_led_values(led_frames, params.get("DIMNESS", 6))
    n = len(colours)
    size = params.get("PREVIEW_SIZE", 400)
    index_image = led_index_image(load_led_centres(), size)

    frame_ids = params.get(
        "PREVIEW_FRAMES", [round(k * n / 4) for k in range(4)]
    )
    panels = render_frames(colours[frame_ids], index_image)
    image = Image.fromarray(np.concatenate(list(panels), axis=1))
    draw = ImageDraw.Draw(image)
    for k, i in enumerate(frame_ids):
        draw.text((k * size + 5, 2), f"frame {i}", fill=(160, 160, 160))
    image_path = params.get(
        "PREVIEW_IMAGE",
        str(output_path.with_name(output_path.stem + "_preview.png")),
    )
    image.save(image_path)
    print(f"Saved preview of frames {frame_ids} to '{image_path}'.")

    if "PREVIEW_VIDEO" in params:
        fps = params.get("PREVIEW_FPS", 24)
        # One frame at a time, to keep memory use low for long sequences
        bgr_frames = (
            render_frames(colours[i : i + 1], index_image)[0, ..., ::-1]
            for i in range(n)
        )
        write_video(bgr_frames, fps, params["PREVIEW_VIDEO"])
        print(
            f"Saved {n}-frame preview video ({n / fps:.1f}s at {fps:g} "
            f"fps) to '{params['PREVIEW_VIDEO']}'."
        )
