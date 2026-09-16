"""Detector and pose models — YOLOX and RTMPose, run with onnxruntime on the CPU.

Both are Apache-2.0, weights included, so the pipeline can ship in a
commercial product (the ultralytics stack it replaces is AGPL-3.0):

* **YOLOX** (Megvii, https://github.com/Megvii-BaseDetection/YOLOX) — the
  official COCO checkpoints exported to ONNX. One pass gives the people *and*
  the COCO objects (handbag, backpack, bottle, cell phone, suitcase…), so the
  held-object stage is free when it only needs COCO classes.
* **RTMPose** (OpenMMLab, https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
  — top-down pose: 17 COCO keypoints per person box, a few ms per crop. The
  SimCC decoding and the crop affine follow the reference implementation in
  rtmlib (Apache-2.0, https://github.com/Tau-J/rtmlib).

Weights are fetched on first use into ``data/models/`` (``python
vision/detector.py --fetch`` does it ahead of time, e.g. in the Docker build).
A model spec is ``"<detector>+<pose>"`` or just ``"<detector>"`` (boxes only,
the floor point is then always the box bottom): ``yolox_s+rtmpose-m`` is the
default, ``yolox_tiny+rtmpose-s`` about twice as fast and less accurate,
``yolox_m+rtmpose-m`` more accurate.
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"

_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
_RTMPOSE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
_RTMPOSE_MIRROR = "https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/"

MODELS: dict[str, dict] = {
    # name: url, network input (w, h), rough COCO AP
    "yolox_nano": {"url": _YOLOX + "yolox_nano.onnx", "input": (416, 416), "ap": 25.8},
    "yolox_tiny": {"url": _YOLOX + "yolox_tiny.onnx", "input": (416, 416), "ap": 32.8},
    "yolox_s": {"url": _YOLOX + "yolox_s.onnx", "input": (640, 640), "ap": 40.5},
    "yolox_m": {"url": _YOLOX + "yolox_m.onnx", "input": (640, 640), "ap": 46.9},
    "yolox_l": {"url": _YOLOX + "yolox_l.onnx", "input": (640, 640), "ap": 49.7},
    "rtmpose-s": {"url": _RTMPOSE + "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
                  "input": (192, 256)},
    "rtmpose-m": {"url": _RTMPOSE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
                  "input": (192, 256)},
    "rtmpose-x": {"url": _RTMPOSE + "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
                  "input": (288, 384)},
}
DEFAULT_MODEL = "yolox_s+rtmpose-m"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
PERSON_CLASS = 0
# Keypoint confidence to count as "seen". RTMPose's SimCC scores run lower
# than heatmap models' — 0.3 here separates visible from hidden joints about
# where 0.5 did for YOLO-pose (mmpose's own default is 0.3 too).
KP_VISIBLE = 0.3


# ── weights ───────────────────────────────────────────────────────────────
def model_path(name: str) -> Path:
    return MODEL_DIR / f"{name}.onnx"


def ensure_model(name: str) -> Path:
    """Local ONNX file for ``name``, downloaded (and unzipped) on first use."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    path = model_path(name)
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = MODELS[name]["url"]
    urls = [url] + ([url.replace(_RTMPOSE, _RTMPOSE_MIRROR)] if url.startswith(_RTMPOSE) else [])
    last = None
    for u in urls:
        try:
            print(f"downloading {name} from {u}")
            with urllib.request.urlopen(u, timeout=60) as r:
                data = r.read()
            break
        except Exception as e:                            # noqa: BLE001
            last = e
    else:
        raise RuntimeError(f"could not download {name}: {last}")
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(n for n in z.namelist() if n.endswith(".onnx"))
            with z.open(member) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        path.write_bytes(data)
    return path


def parse_model_spec(spec: str) -> tuple[str, str | None]:
    """``"yolox_s+rtmpose-m"`` → ("yolox_s", "rtmpose-m"); ``"yolox_s"`` → ("yolox_s", None)."""
    det, _, pose = (spec or DEFAULT_MODEL).partition("+")
    det, pose = det.strip(), pose.strip() or None
    if not det.startswith("yolox"):
        raise ValueError(f"detector must be a yolox_* model, not {det!r}")
    if pose is not None and not pose.startswith("rtmpose"):
        raise ValueError(f"pose model must be an rtmpose-* model, not {pose!r}")
    return det, pose


def _session(path: Path, device: str):
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        providers = ["CUDAExecutionProvider"] + providers
    return ort.InferenceSession(str(path), providers=providers)


# ── YOLOX ─────────────────────────────────────────────────────────────────
class Yolox:
    """``det(frame, conf)`` → (boxes xyxy (N,4), scores (N,), class ids (N,)),
    class-aware NMS applied. The frame is letter-boxed to the network input,
    so ``imgsz`` is a property of the weights (416 for tiny/nano, 640 else)."""

    def __init__(self, name: str = "yolox_s", device: str = "cpu", nms: float = 0.45) -> None:
        self.name = name
        self.input_w, self.input_h = MODELS[name]["input"]
        self.nms = nms
        self.session = _session(ensure_model(name), device)
        self.input_name = self.session.get_inputs()[0].name
        self.names = COCO_CLASSES
        self._grid, self._stride = self._grids()

    def _grids(self) -> tuple[np.ndarray, np.ndarray]:
        grids, strides = [], []
        for s in (8, 16, 32):
            ys, xs = np.meshgrid(np.arange(self.input_h // s), np.arange(self.input_w // s), indexing="ij")
            g = np.stack((xs, ys), -1).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((len(g), 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def __call__(self, frame: np.ndarray, conf: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        r = min(self.input_h / h, self.input_w / w)
        resized = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized     # BGR, 0–255: what the export expects
        x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32))
        out = self.session.run(None, {self.input_name: x})[0][0]   # (anchors, 85)

        xy = (out[:, :2] + self._grid) * self._stride
        wh = np.exp(out[:, 2:4]) * self._stride
        cls = out[:, 5:].argmax(1)
        scores = out[:, 4] * out[:, 5:][np.arange(len(out)), cls]
        keep = scores >= conf
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        xy, wh, cls, scores = xy[keep], wh[keep], cls[keep], scores[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        idx = nms_boxes(boxes, scores, self.nms, cls)
        return boxes[idx].astype(np.float32), scores[idx].astype(np.float32), cls[idx].astype(int)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, thresh: float, cls: np.ndarray | None = None) -> np.ndarray:
    """Indices kept by NMS; with ``cls`` boxes of different classes never
    suppress each other (class-aware), without it they do (agnostic)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    b = boxes.astype(np.float32).copy()
    if cls is not None:                                   # offset each class into its own space
        off = (b[:, 2:].max() + 1) * cls.astype(np.float32)[:, None]
        b += off
    xywh = np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(np.float32).tolist(), 0.0, thresh)
    return np.asarray(idx, dtype=int).reshape(-1)


# ── RTMPose ───────────────────────────────────────────────────────────────
class RtmPose:
    """``pose(frame, boxes)`` → (keypoints (N,17,2) in frame pixels, scores
    (N,17)). Top-down: each box is padded by 1.25, warped to the network's
    aspect ratio and decoded from the SimCC x/y distributions; all crops of
    one frame go through the network as one batch."""

    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)
    PADDING = 1.25
    SIMCC_SPLIT = 2.0

    def __init__(self, name: str = "rtmpose-m", device: str = "cpu") -> None:
        self.name = name
        self.input_w, self.input_h = MODELS[name]["input"]
        self.session = _session(ensure_model(name), device)
        self.input_name = self.session.get_inputs()[0].name

    def _crop(self, frame: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        bw, bh = (x2 - x1) * self.PADDING, (y2 - y1) * self.PADDING
        aspect = self.input_w / self.input_h
        if bw > bh * aspect:
            scale = np.array([bw, bw / aspect], np.float32)
        else:
            scale = np.array([bh * aspect, bh], np.float32)
        sx, sy = self.input_w / scale[0], self.input_h / scale[1]
        m = np.array([[sx, 0, self.input_w / 2 - center[0] * sx],
                      [0, sy, self.input_h / 2 - center[1] * sy]], np.float32)
        crop = cv2.warpAffine(frame, m, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)   # trained with bgr_to_rgb
        return (crop - self.MEAN) / self.STD, center, scale

    def __call__(self, frame: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        crops, centers, scales = zip(*(self._crop(frame, b) for b in boxes))
        x = np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2))
        simcc_x, simcc_y = self.session.run(None, {self.input_name: x})
        n, k = simcc_x.shape[:2]
        lx, ly = simcc_x.argmax(2), simcc_y.argmax(2)
        vals = np.minimum(simcc_x.max(2), simcc_y.max(2))
        locs = np.stack((lx, ly), -1).astype(np.float32) / self.SIMCC_SPLIT
        centers, scales = np.stack(centers), np.stack(scales)
        kp = locs / np.array([self.input_w, self.input_h], np.float32) * scales[:, None] \
            + centers[:, None] - scales[:, None] / 2
        kp[vals <= 0] = -1
        return kp.astype(np.float32), vals.astype(np.float32)


# ── drawing (replaces ultralytics' results.plot()) ────────────────────────
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def colour(i: int) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_people(canvas: np.ndarray, people: list[dict], kp_thresh: float = KP_VISIBLE) -> np.ndarray:
    """Boxes, ids, skeletons and floor points for ``extract_people`` dicts."""
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        col = colour(p["raw_id"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.putText(canvas, f"#{p['raw_id']} {p['conf']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        kp = p.get("kp")
        if kp:
            for a, b in SKELETON:
                if kp[a][2] >= kp_thresh and kp[b][2] >= kp_thresh:
                    cv2.line(canvas, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), col, 1)
            for x, y, c in kp:
                if c >= kp_thresh:
                    cv2.circle(canvas, (int(x), int(y)), 2, col, -1)
        fx, fy = (int(v) for v in p["foot"])
        cv2.circle(canvas, (fx, fy), 4, col, -1 if p["foot_src"] == "box" else 1)
    return canvas


def draw_objects(canvas: np.ndarray, objects: list[dict]) -> np.ndarray:
    for o in objects:
        col = colour(o["person_id"]) if o["person_id"] != "" else (200, 200, 200)
        x1, y1, x2, y2 = int(o["x1"]), int(o["y1"]), int(o["x2"]), int(o["y2"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
        cv2.putText(canvas, f"{o['label']} {o['conf']:.2f}", (x1, min(canvas.shape[0] - 2, y2 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="fetch model weights into data/models")
    ap.add_argument("--fetch", nargs="*", default=None,
                    help=f"model names to download (default: the ones in {DEFAULT_MODEL})")
    a = ap.parse_args()
    names = a.fetch if a.fetch else [n for n in parse_model_spec(DEFAULT_MODEL) if n]
    for n in names:
        print(n, "→", ensure_model(n))


if __name__ == "__main__":
    main()
