import numpy as np

from gen_video_frames.led_frames import scale_led_values
from gen_video_frames.video import crop_frame


def test_crop_frame():
    frame = np.zeros((360, 640, 3), dtype="uint8")
    cropped = crop_frame(frame, x_frac=(0.23, 0.77), y_frac=(0.0, 0.96))
    assert cropped.shape == (346, 346, 3)


def test_scale_led_values():
    z = np.array([0, 16, 128, 255])
    assert scale_led_values(z, dimness=6).tolist() == [0, 0, 10, 42]
    assert scale_led_values(z, dimness=1).tolist() == [0, 1, 64, 254]
