import numpy as np

from gen_video_frames.led_frames import scale_led_values
from gen_video_frames.preview import (
    led_index_image,
    load_led_centres,
    render_frames,
    unscale_led_values,
)


def test_load_led_centres():
    centres = load_led_centres()
    assert centres.shape == (1593, 2)


def test_every_led_is_drawn():
    index = led_index_image(load_led_centres(), 400)
    assert set(np.unique(index)) == set(range(-1, 1593))


def test_render_frames():
    index = np.array([[-1, 0], [1, 1]])
    colours = np.array([[[10, 20, 30], [40, 50, 60]]], dtype="uint8")
    image = render_frames(colours, index)
    assert image.shape == (1, 2, 2, 3)
    assert image[0, 0, 0].tolist() == [0, 0, 0]
    assert image[0, 1, 1].tolist() == [40, 50, 60]


def test_unscale_led_values():
    # raw = sqrt(z * 256 * dimness), inverting scale_led_values()
    z = np.array([0, 6, 42])
    assert unscale_led_values(z, 6).tolist() == [0, 96, 254]
    assert scale_led_values(np.array([96, 254]), 6).tolist() == [6, 42]
