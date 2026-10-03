"""
Command-line interface:

    gen-video-frames download PARAMS_FILE     # VIDEO_URL clip -> VIDEO_FILE
    gen-video-frames find-loops PARAMS_FILE   # loop search over VIDEO_FILE
    gen-video-frames generate PARAMS_FILE     # LED frames -> OUTPUT_FILE
    gen-video-frames preview PARAMS_FILE      # OUTPUT_FILE as displayed

See README.md for the parameters file keys.
"""

import argparse

from gen_video_frames.params import load_params


def download(params):
    from gen_video_frames.video import (
        download_youtube_video,
        get_video_info,
        print_video_info,
    )

    print_video_info(get_video_info(params["VIDEO_URL"]))
    start_time = params.get("START_TIME")
    duration = params.get("DURATION")
    end_time = None if duration is None else (start_time or 0) + duration
    print(f"\nDownloading to '{params['VIDEO_FILE']}'...")
    download_youtube_video(
        params["VIDEO_URL"],
        output_filename=params["VIDEO_FILE"],
        start_time=start_time,
        end_time=end_time,
    )


def find_loops(params):
    from gen_video_frames.loops import find_loops

    find_loops(params)


def generate(params):
    from gen_video_frames.led_frames import generate_output

    generate_output(params)


def preview(params):
    from gen_video_frames.preview import preview

    preview(params)


COMMANDS = {
    "download": download,
    "find-loops": find_loops,
    "generate": generate,
    "preview": preview,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("params_file", help="YAML parameters file")
    args = parser.parse_args()
    COMMANDS[args.command](load_params(args.params_file))


if __name__ == "__main__":
    main()
