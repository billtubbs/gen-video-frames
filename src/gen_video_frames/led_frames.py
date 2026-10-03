"""
Converting video frames or images into LED data for the 1593-LED display.

The conversion itself (each LED's average over its patch of a 256x256
image) is display1593.image_conversion's prepare_image()/convert_image().
"""

from pathlib import Path

import cv2
import numpy as np
from display1593.image_conversion import convert_image, prepare_image
from PIL import Image

from gen_video_frames.video import crop_frame


def scale_led_values(z, dimness=6):
    """
    Gamma-corrects and dims raw convert_image() output (0-255) into the
    values sent to the display.
    """
    z = z.astype(float) ** 2 / (256 * dimness)
    return z.astype("uint8")


def image_to_led_frame(image, dimness=6):
    """
    Converts a source image into 1593 gamma-corrected LED RGB values, ready
    to send directly to the display.
    """
    return scale_led_values(convert_image(prepare_image(image)), dimness)


def generate_led_frames(image_dir, output_dir, dimness=6, pattern="*.jpg"):
    """
    Converts every image in image_dir to a 1593x3 LED RGB frame and saves
    each as an integer CSV file (one row per LED) in output_dir.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob(pattern))
    print(f"Converting {len(image_paths)} images from '{image_dir}'...")
    for image_path in image_paths:
        image = Image.open(image_path)
        z = image_to_led_frame(image, dimness=dimness)
        csv_path = output_dir / (image_path.stem + ".csv")
        np.savetxt(csv_path, z, fmt="%d", delimiter=",")

    print(f"Saved {len(image_paths)} LED frame CSVs to '{output_dir}'.")
    return image_paths


def compute_led_frames(video_path, x_frac, y_frac):
    """
    Reads every frame of a video in order, crops it and converts it to
    1593 raw RGB LED values (before gamma/dimness).

    :return: (led_frames, fps), led_frames shape (n_frames, 1593, 3), uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file '{video_path}'")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Converting {n_total} frames of '{video_path}' to LED values...")
    led_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = crop_frame(frame, x_frac=x_frac, y_frac=y_frac)
        image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        led_frames.append(convert_image(prepare_image(image)))
        if len(led_frames) % 1800 == 0:
            print(f"  {len(led_frames)} / {n_total}")
    cap.release()
    return np.array(led_frames, dtype="uint8"), fps


def load_or_compute_led_frames(params):
    """
    Raw LED values for every frame of params["VIDEO_FILE"], cropped to
    CROP_X_FRAC/CROP_Y_FRAC, cached in params["LED_FRAMES_FILE"]. The cache
    isn't checked against the parameters, so delete it (or use a new
    LED_FRAMES_FILE name) after changing the video or crop.

    :return: (led_frames, fps), led_frames shape (n_frames, 1593, 3), uint8.
    """
    path = Path(params["LED_FRAMES_FILE"])
    if path.exists():
        print(f"Loading cached LED frames from '{path}'")
        data = np.load(path)
        return data["led_frames"], float(data["fps"])
    led_frames, fps = compute_led_frames(
        params["VIDEO_FILE"],
        params.get("CROP_X_FRAC", [0.0, 1.0]),
        params.get("CROP_Y_FRAC", [0.0, 1.0]),
    )
    np.savez_compressed(path, led_frames=led_frames, fps=fps)
    print(f"Saved LED frames to '{path}'")
    return led_frames, fps


def generate_output(params):
    """
    Writes the display LED data for frames START_FRAME to END_FRAME - 1
    (default: all frames) to OUTPUT_FILE, a compressed .npz with array
    "led_frames", shape (n_frames, 1593, 3), uint8, plus the source "fps".
    """
    led_frames, fps = load_or_compute_led_frames(params)
    start = params.get("START_FRAME", 0)
    end = params.get("END_FRAME", len(led_frames))
    frames = scale_led_values(led_frames[start:end], params.get("DIMNESS", 6))

    output_path = params["OUTPUT_FILE"]
    np.savez_compressed(output_path, led_frames=frames, fps=fps)
    print(
        f"Saved {len(frames)} LED frames ({len(frames) / fps:.2f}s at "
        f"{fps:.0f} fps, source frames {start}-{end - 1}) to "
        f"'{output_path}'."
    )
    return frames, fps
