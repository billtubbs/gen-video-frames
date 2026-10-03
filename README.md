# gen-video-frames

Generates LED frame data for the
[display1593](https://github.com/billtubbs/display1593) 1593-LED display
from a YouTube video or a local video file. Everything about a frame
sequence (source, clip range, crop, loop search, output) is specified in
one YAML parameters file.

## Setup

Requires `ffmpeg` on the PATH (used by yt-dlp to trim clips) and Python
3.10+.

```
python3 -m venv .venv
.venv/bin/pip install -e .            # or -e '.[test]' for pytest
.venv/bin/pip install -e /path/to/display1593 --no-deps
```

`display1593` provides the image-to-LED conversion
(`display1593.image_conversion`, numpy/Pillow only, plus the LED mask
data). It's installed with `--no-deps` to avoid its hardware-only
dependencies (`serial`, `numba`, `serial_comm`), and isn't listed in
`pyproject.toml` because it isn't on PyPI.

To use the package from another project (e.g.
[led-fire-place](https://github.com/billtubbs/led-fire-place)), install it
into that project's environment the same way:
`pip install -e /path/to/gen-video-frames`.

## Usage

```
gen-video-frames download PARAMS_FILE     # VIDEO_URL clip -> VIDEO_FILE
gen-video-frames find-loops PARAMS_FILE   # loop search over VIDEO_FILE
gen-video-frames generate PARAMS_FILE     # LED frames -> OUTPUT_FILE
gen-video-frames preview PARAMS_FILE      # OUTPUT_FILE as displayed
```

A typical workflow: `download` a clip, `find-loops` to choose a loop that
joins smoothly, set `START_FRAME`/`END_FRAME` to it, `generate` the output file,
`preview` it, and copy it to the display (`fireplace/data/fire_frames.npz`
for `fireplace/play_fire_frames.py`). For a local video, skip `download`.

`find-loops` and `generate` first convert every frame of `VIDEO_FILE` to
LED values (about 5 ms per frame) and cache them in `LED_FRAMES_FILE`, so
later runs are quick. The cache isn't checked against the parameters:
use a new `LED_FRAMES_FILE` name, or delete it, after changing the video
or crop.

`preview` draws each LED as a dot at its real position on the display,
coloured as it will show (the output with the gamma/dimness scaling
undone), so a sequence can be checked before copying it to the Pi. The
LED positions are packaged in `gen_video_frames/data/led_centres_1593.csv`
(exported from display1593's `ledArray_data_1593.py`).

Examples:
- [examples/params.yaml](examples/params.yaml) - the LED fireplace, from
  a YouTube video.
- [examples/phytoBlue.yaml](examples/phytoBlue.yaml) - a rotating globe,
  from a local NASA video.

## Parameters file keys

| Key | Used by | Description |
|---|---|---|
| `VIDEO_URL` | download | YouTube URL. Downloaded as format 18 (640x360) - plenty for 1593 LEDs, and available without a PO token. |
| `START_TIME`, `DURATION` | download | Clip range in seconds (default: whole video). |
| `VIDEO_FILE` | all | Downloaded clip, or a local source video. |
| `CROP_X_FRAC`, `CROP_Y_FRAC` | find-loops, generate | Region of each frame mapped to the display, as `[min, max]` fractions of width/height (default `[0.0, 1.0]`). Centre-cropped to a square afterwards. |
| `LED_FRAMES_FILE` | find-loops, generate | Cache of raw LED values for every frame (`.npz`). |
| `MIN_LOOP_SECONDS`, `MAX_LOOP_SECONDS` | find-loops | Loop length range to search. |
| `SEAM_HALF_WINDOW_SECONDS` | find-loops | Seams are scored over this long either side of the join, so motion has to match too (default 0.5). |
| `MIN_CANDIDATE_SEPARATION_SECONDS` | find-loops | Candidates must differ by at least this much in start or end (default 5). |
| `N_CANDIDATES` | find-loops | Number of candidates to report (default 10). |
| `LOOP_CANDIDATES_FILE` | find-loops | CSV of candidates (default `loop_candidates.csv`). |
| `LOOP_SEAMS_DIR` | find-loops | A short video of the cropped source around each candidate's seam (default `loop_seams`). |
| `START_FRAME`, `END_FRAME` | generate | Frames of `VIDEO_FILE` to output, end exclusive (default: all). |
| `DIMNESS` | generate | Brightness divisor; output = raw<sup>2</sup> / (256 x `DIMNESS`) (default 6). |
| `OUTPUT_FILE` | generate, preview | Output `.npz`: array `led_frames`, shape `(n_frames, 1593, 3)`, `uint8`, plus the source `fps`. |
| `PREVIEW_FRAMES` | preview | Output frame indices to show side by side (default: 4 evenly spaced). |
| `PREVIEW_SIZE` | preview | Size of each frame's image in pixels (default 400). |
| `PREVIEW_IMAGE` | preview | Preview image (default: `OUTPUT_FILE` name + `_preview.png`). |
| `PREVIEW_VIDEO`, `PREVIEW_FPS` | preview | Optional preview video of every frame, at `PREVIEW_FPS` (default 24). |

## Loop search

A loop from frame t to t + g (played t, ..., t + g - 1, then back to t) is
scored by the RMS difference between the 1593-LED frames either side of
the seam (frames t - k ... t + k - 1 vs. t + g - k ... t + g + k - 1), on
the 0-255 raw LED scale. Only the band of the frame-to-frame distance
matrix for the requested loop lengths is computed, so a 10-minute clip
(18000 frames) takes seconds once the LED values are cached.

`find-loops` also prints the median and 90th percentile RMS difference
between consecutive frames: a seam scoring near or below the median
should be hard to see. Candidates with exactly the same length are a sign
that the source video is itself a loop.

## Modules

- `gen_video_frames.video` - `get_video_info`/`print_video_info`,
  `download_youtube_video` (yt-dlp, optionally a time range),
  `extract_frames_at_fps`, `extract_frames`, `crop_frame`,
  `sample_and_crop_youtube_clip`, `write_video` (frames -> H.264 .mp4 via
  ffmpeg).
- `gen_video_frames.led_frames` - `compute_led_frames` (video -> raw LED
  values), `load_or_compute_led_frames` (cached), `scale_led_values`
  (gamma/dimness), `image_to_led_frame`, `generate_led_frames` (image
  folder -> one CSV per frame), `generate_output`.
- `gen_video_frames.loops` - `distance_band`, `seam_scores`,
  `best_candidates`, `save_seam_preview`, `find_loops`.
- `gen_video_frames.preview` - `load_led_centres`, `led_index_image`,
  `render_frames`, `unscale_led_values`, `preview`.
- `gen_video_frames.params` - `load_params`.

## Tests

```
pytest
```
