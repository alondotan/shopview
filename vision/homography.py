"""Camera → store-map homography.

A homography is the 3x3 matrix that maps points on one plane to another. The
shop floor is a plane, so the bottom-centre ("foot") point of a person's box in
the camera image maps to a point on the store map — which is what turns pixel
detections into store coordinates.

Calibration is done by clicking >= 4 matching point pairs in the /calibrate
page; this module fits the matrix and applies it to a tracks CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "data" / "calibration"
TRACK_DIR = ROOT / "data" / "tracks"


def solve_homography(img_pts: list, map_pts: list) -> dict:
    """Least-squares fit of the camera→map homography.

    Returns the matrix plus the per-point reprojection error in map pixels, so
    the UI can show which clicked pair is off.
    """
    import cv2

    src = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(map_pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(src) < 4:
        raise ValueError("at least 4 point pairs are required")

    # method=0 → plain least squares over every pair (no RANSAC: with a handful
    # of hand-clicked points we want all of them to count).
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("could not fit a homography — are the points collinear?")

    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)

    return {
        "H": H.tolist(),
        "errors_px": [round(float(e), 2) for e in errors],
        "rms_px": round(float(np.sqrt((errors ** 2).mean())), 2),
        "max_px": round(float(errors.max()), 2),
    }


def load_calibration(video_id: str) -> dict:
    path = CALIB_DIR / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — calibrate at /calibrate first")
    return json.loads(path.read_text(encoding="utf-8"))


def to_map(H: list | np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply the homography to an (N,2) array of image points."""
    import cv2

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def warp_tracks(video_id: str, tracks_csv: Path | None = None,
                out_csv: Path | None = None) -> Path:
    """Add map_x / map_y columns to a tracks CSV using the saved calibration."""
    calib = load_calibration(video_id)

    if tracks_csv is None:
        candidates = sorted(TRACK_DIR.glob(f"{video_id}*_tracks.csv"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"no tracks CSV for {video_id}")
        tracks_csv = candidates[0]

    rows = list(csv.DictReader(tracks_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{tracks_csv} is empty")

    feet = np.array([[float(r["foot_x"]), float(r["foot_y"])] for r in rows])
    mapped = to_map(calib["H"], feet)

    out_csv = out_csv or tracks_csv.with_name(tracks_csv.stem + "_map.csv")
    fieldnames = list(rows[0].keys()) + ["map_x", "map_y"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row, (mx, my) in zip(rows, mapped):
            w.writerow({**row, "map_x": round(float(mx), 1), "map_y": round(float(my), 1)})

    print(f"{len(rows)} rows → {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_id", help="e.g. KMJS66jBtVQ")
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    warp_tracks(args.video_id, args.tracks, args.out)


if __name__ == "__main__":
    main()
