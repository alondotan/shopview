"""Download a source video (YouTube or any yt-dlp supported URL) to data/videos/."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"


def download(url: str, out_dir: Path = VIDEO_DIR, max_height: int = 720) -> Path:
    """Download `url` as a single mp4 file. Returns the local path."""
    try:
        import yt_dlp
    except ImportError:
        sys.exit("yt-dlp is not installed — run: pip install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        # H.264 (avc1) first: it decodes everywhere — OpenCV builds without an AV1
        # decoder (the Linux container) and every browser. Fall back to any mp4.
        "format": (f"bestvideo[height<={max_height}][vcodec^=avc1]+bestaudio[ext=m4a]"
                   f"/bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
                   f"/best[height<={max_height}]"),
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    print(f"\nTitle    : {info.get('title')}")
    print(f"Duration : {info.get('duration')}s")
    print(f"Saved to : {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="video URL")
    ap.add_argument("--max-height", type=int, default=720)
    args = ap.parse_args()
    download(args.url, max_height=args.max_height)


if __name__ == "__main__":
    main()
