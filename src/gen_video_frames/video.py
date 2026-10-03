"""Downloading, sampling and cropping video."""

import os
import subprocess

import cv2
import numpy as np
import yt_dlp


def get_video_info(url):
    """
    Fetches metadata about a YouTube video without downloading it, for the
    same format that download_youtube_video() would select.

    :param url: YouTube video URL.
    :return: yt-dlp info dict (title, width, height, fps, duration, ext, ...).
    """
    ydl_opts = {
        "format": "18/mp4/best",
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def print_video_info(info):
    """Prints a video's resolution, fps, duration and format."""
    duration = info.get("duration")
    if duration is not None:
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
    else:
        duration_str = "unknown"
    print(f"Title:      {info.get('title')}")
    print(f"Resolution: {info.get('width')}x{info.get('height')}")
    print(f"Frame rate: {info.get('fps')} fps")
    print(f"Duration:   {duration_str} ({duration}s)")
    print(f"Format:     {info.get('format')} [{info.get('ext')}]")


def download_youtube_video(
    url, output_filename="video.mp4", start_time=None, end_time=None
):
    """
    Downloads a YouTube video using yt-dlp, or just a clipped section of it
    if start_time/end_time are given.

    Requires ffmpeg to be installed (yt-dlp uses it to cut the clip so only
    the requested section is downloaded, not the whole video).

    :param url: YouTube video URL.
    :param output_filename: Path to save the downloaded video/clip.
    :param start_time: Clip start time in seconds (None = start of video).
    :param end_time: Clip end time in seconds (required if start_time is
        given).
    """
    print(f"Downloading video from {url}...")
    ydl_opts = {
        # YouTube now forces SABR streaming (no direct URL) for the default
        # "tv" client, and the "android" client needs a PO token for most
        # formats. "18" (360p mp4) is reliably available on the "android"/
        # "web" clients without one. See yt-dlp issue #12482.
        "format": "18/mp4/best",
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "outtmpl": output_filename,
        # Without this, yt-dlp silently reuses an existing file at
        # output_filename instead of re-downloading, even if start_time/
        # end_time have changed since it was last written.
        "overwrites": True,
    }
    if start_time is not None or end_time is not None:
        if end_time is None:
            raise ValueError(
                "end_time must be given when start_time is given."
            )
        clip_start = start_time or 0.0
        ydl_opts["download_ranges"] = lambda info, ydl: [
            {"start_time": clip_start, "end_time": end_time}
        ]
        ydl_opts["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete.")
    return output_filename


def extract_frames(video_path, output_dir="frames", frame_skip=30):
    """
    Extracts frames from a local video file.

    :param video_path: Path to the video file.
    :param output_dir: Directory where image frames will be saved.
    :param frame_skip: Extract every 'N' frames (e.g., 30 extracts roughly
        1 frame per second on a 30fps video).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    print(f"Extracting frames to '{output_dir}'...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Extract frame based on skipped interval
        if count % frame_skip == 0:
            frame_filename = os.path.join(
                output_dir, f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Extraction complete. Total frames saved: {saved_count}")


def extract_frames_at_fps(video_path, fps, start_time=0.0, duration=None):
    """
    Samples frames from a local video file at a fixed target frame rate,
    by seeking to each target timestamp rather than decoding every frame.

    :param video_path: Path to the video file.
    :param fps: Number of frames to sample per second of video.
    :param start_time: Offset (seconds) into the video to start sampling.
    :param duration: How many seconds to sample (None = to end of video).
    :return: List of frames (numpy arrays, BGR order as returned by OpenCV).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file '{video_path}'")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not video_fps:
        raise ValueError(f"Could not determine frame rate of '{video_path}'")

    if duration is None:
        duration = total_frames / video_fps - start_time
    n_samples = round(duration * fps)

    print(f"Sampling {n_samples} frames at {fps} fps from '{video_path}'...")
    video_duration = total_frames / video_fps
    if start_time + duration > video_duration + 1e-6:
        print(
            f"Warning: requested {start_time:.3f}-"
            f"{start_time + duration:.3f}s "
            f"but '{video_path}' is only {video_duration:.3f}s long. Frames "
            "past the end will repeat the last frame."
        )

    frames = []
    for i in range(n_samples):
        t = start_time + i / fps
        frame_index = min(round(t * video_fps), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(
                f"Warning: could not read frame at t={t:.3f}s "
                f"(index {frame_index})"
            )
            continue
        frames.append(frame)

    cap.release()
    print(f"Sampled {len(frames)} frames.")
    return frames


def crop_frame(frame, x_frac=(0.0, 1.0), y_frac=(0.0, 1.0)):
    """
    Crops a frame (as returned by OpenCV, shape (H, W, C)) to a rectangular
    region given as fractions of the full width/height.

    :param frame: Image array.
    :param x_frac: (x_min, x_max) fractions of width, each in [0, 1].
    :param y_frac: (y_min, y_max) fractions of height, each in [0, 1].
    """
    height, width = frame.shape[:2]
    x1, x2 = round(x_frac[0] * width), round(x_frac[1] * width)
    y1, y2 = round(y_frac[0] * height), round(y_frac[1] * height)
    return frame[y1:y2, x1:x2]


def sample_and_crop_youtube_clip(
    url,
    start_time,
    duration,
    fps=16,
    x_frac=(0.0, 1.0),
    y_frac=(0.0, 1.0),
    video_filename="clip.mp4",
    output_dir="frames",
):
    """
    Downloads a short clip from a YouTube video, samples frames from it at
    a fixed rate, crops each frame to a rectangular region, and saves them.

    :param url: YouTube video URL.
    :param start_time: Clip start time in the source video, in seconds.
    :param duration: Clip duration in seconds.
    :param fps: Number of frames to sample per second.
    :param x_frac: (x_min, x_max) crop fractions of width, each in [0, 1].
    :param y_frac: (y_min, y_max) crop fractions of height, each in [0, 1].
    :param video_filename: Path to save the downloaded clip.
    :param output_dir: Directory where cropped frame images will be saved.
    """
    video_path = download_youtube_video(
        url,
        output_filename=video_filename,
        start_time=start_time,
        end_time=start_time + duration,
    )

    # The downloaded clip starts at t=0, regardless of start_time in the
    # source video, since yt-dlp already trimmed it.
    frames = extract_frames_at_fps(
        video_path, fps=fps, start_time=0.0, duration=duration
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    for i, frame in enumerate(frames):
        cropped = crop_frame(frame, x_frac=x_frac, y_frac=y_frac)
        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_filename, cropped)
        saved_paths.append(frame_filename)

    print(f"Saved {len(saved_paths)} cropped frames to '{output_dir}'.")
    return saved_paths


def write_video(frames, fps, path):
    """
    Writes BGR frames (any iterable of same-size arrays, so long videos
    needn't be held in memory) to an H.264 .mp4 with ffmpeg. (OpenCV's own
    VideoWriter can't write .mp4 in some builds, and fails silently.)
    """
    process = None
    for frame in frames:
        if process is None:
            height, width = frame.shape[:2]
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
                # H.264/yuv420p needs even dimensions
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ]  # fmt: skip
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    if process is None:
        raise ValueError("No frames to write")
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing '{path}'")
