"""ByteTrack — multi-object tracking by associating every detection box.

A compact port of the reference implementation (Zhang et al., ECCV 2022,
https://github.com/ifzhang/ByteTrack, MIT licence): a constant-velocity Kalman
filter per track, and a two-stage association per frame — high-confidence
detections are matched to tracks by IoU first; the leftover tracks then get a
second chance against the *low*-confidence detections, which is what keeps a
half-hidden shopper on the same id instead of dropping the track and starting
a new one when they reappear.

Thresholds (see ``detect_people.tracker_config``):

* ``track_high``  — a detection at or above this joins the first association
* ``track_low``   — below this a detection is ignored altogether
* ``new_track``   — an unmatched detection needs this much to start a track
* ``buffer``      — analysed frames a lost track is kept before it is dropped
* ``match_thresh``— max IoU distance (1 − IoU) for the first association
* ``fuse_score``  — weight the IoU by the detection score in the first stage

``update()`` takes the frame's boxes and returns the active tracks with the
index of the detection each one matched, so the caller can attach whatever
else the detector produced (keypoints, class) to the track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import lap
except ImportError:                                   # pragma: no cover
    lap = None


# ── Kalman filter (8-dim state: x, y, aspect, height and their velocities) ──
class KalmanFilter:
    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def multi_predict(self, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = means[:, 3]
        std = np.stack([self._std_pos * h, self._std_pos * h, 1e-2 * np.ones_like(h),
                        self._std_pos * h, self._std_vel * h, self._std_vel * h,
                        1e-5 * np.ones_like(h), self._std_vel * h], axis=1)
        motion_cov = np.array([np.diag(s) for s in np.square(std)])
        means = means @ self._motion_mat.T
        covs = self._motion_mat @ covs @ self._motion_mat.T + motion_cov
        return means, covs

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        cov = self._update_mat @ cov @ self._update_mat.T
        return mean, cov + innovation_cov

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        gain = np.linalg.solve(proj_cov, (cov @ self._update_mat.T).T).T
        new_mean = mean + (xyah - proj_mean) @ gain.T
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov


# ── tracks ────────────────────────────────────────────────────────────────
NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class STrack:
    def __init__(self, xyxy, score: float, det_index: int) -> None:
        x1, y1, x2, y2 = xyxy
        self._tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)
        self.score = float(score)
        self.det_index = det_index
        self.mean = None
        self.cov = None
        self.state = NEW
        self.is_activated = False
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    def activate(self, kf: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.track_id = track_id
        self.mean, self.cov = kf.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, kf: KalmanFilter, det: "STrack", frame_id: int) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, self.tlwh_to_xyah(det._tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score, self.det_index = det.score, det.det_index

    def update(self, kf: KalmanFilter, det: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.cov = kf.update(self.mean, self.cov, self.tlwh_to_xyah(det._tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.score, self.det_index = det.score, det.det_index


# ── matching ──────────────────────────────────────────────────────────────
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N,4 xyxy) against every box in ``b`` (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _iou_distance(tracks: list[STrack], dets: list[STrack]) -> np.ndarray:
    return 1.0 - iou_matrix([t.xyxy for t in tracks], [d.xyxy for d in dets])


def _fuse_score(cost: np.ndarray, dets: list[STrack]) -> np.ndarray:
    if cost.size == 0:
        return cost
    scores = np.array([d.score for d in dets])
    return 1.0 - (1.0 - cost) * scores[None, :]


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Min-cost matching with ``cost > thresh`` forbidden. Returns (matches,
    unmatched rows, unmatched cols)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if lap is not None:
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
        matches = [(i, int(j)) for i, j in enumerate(x) if j >= 0]
        return matches, [int(i) for i in np.where(x < 0)[0]], [int(j) for j in np.where(y < 0)[0]]
    # fallback: greedy on the cheapest pairs (lap is in requirements; this keeps tests running without it)
    matches, used_r, used_c = [], set(), set()
    for r, c in zip(*np.unravel_index(np.argsort(cost, axis=None), cost.shape)):
        if cost[r, c] > thresh:
            break
        if r in used_r or c in used_c:
            continue
        matches.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return (matches, [i for i in range(cost.shape[0]) if i not in used_r],
            [j for j in range(cost.shape[1]) if j not in used_c])


# ── the tracker ───────────────────────────────────────────────────────────
@dataclass
class Track:
    """One active track after ``update()``: the Kalman-smoothed box, the id,
    and which of this frame's detections it came from."""
    id: int
    xyxy: tuple[float, float, float, float]
    score: float
    det_index: int


class ByteTracker:
    def __init__(self, track_high: float = 0.35, track_low: float = 0.1, new_track: float = 0.45,
                 buffer: int = 30, match_thresh: float = 0.8, fuse_score: bool = True) -> None:
        self.track_high, self.track_low, self.new_track = track_high, track_low, new_track
        self.max_time_lost = buffer
        self.match_thresh, self.fuse_score = match_thresh, fuse_score
        self.kf = KalmanFilter()
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self.frame_id = 0
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _predict(self, tracks: list[STrack]) -> None:
        if not tracks:
            return
        means = np.stack([t.mean for t in tracks])
        covs = np.stack([t.cov for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TRACKED:
                means[i, 7] = 0                       # a lost track does not keep growing
        means, covs = self.kf.multi_predict(means, covs)
        for t, m, c in zip(tracks, means, covs):
            t.mean, t.cov = m, c

    def update(self, xyxy, scores) -> list[Track]:
        """One frame. ``xyxy`` (N,4) and ``scores`` (N,) are every detection
        the detector let through (down to ``track_low``)."""
        self.frame_id += 1
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        activated, refind, lost, removed = [], [], [], []

        high = scores >= self.track_high
        second = (scores >= self.track_low) & ~high
        dets = [STrack(b, s, i) for i, (b, s) in enumerate(zip(xyxy, scores)) if high[i]]
        dets_second = [STrack(b, s, i) for i, (b, s) in enumerate(zip(xyxy, scores)) if second[i]]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # 1. high-confidence detections vs. tracked + lost tracks
        pool = _join(tracked, self.lost)
        self._predict(pool)
        dists = _iou_distance(pool, dets)
        if self.fuse_score:
            dists = _fuse_score(dists, dets)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)
        for it, idet in matches:
            track, det = pool[it], dets[idet]
            if track.state == TRACKED:
                track.update(self.kf, det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kf, det, self.frame_id)
                refind.append(track)

        # 2. the leftover tracked tracks vs. the low-confidence detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        dists = _iou_distance(r_tracked, dets_second)
        matches, u_track, _ = linear_assignment(dists, 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], dets_second[idet]
            if track.state == TRACKED:
                track.update(self.kf, det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kf, det, self.frame_id)
                refind.append(track)
        for it in u_track:
            track = r_tracked[it]
            if track.state != LOST:
                track.state = LOST
                lost.append(track)

        # 3. unconfirmed (one-frame-old) tracks vs. what is still unmatched
        dets = [dets[i] for i in u_det]
        dists = _iou_distance(unconfirmed, dets)
        if self.fuse_score:
            dists = _fuse_score(dists, dets)
        matches, u_unconfirmed, u_det = linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(self.kf, dets[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = REMOVED
            removed.append(track)

        # 4. new tracks from confident leftovers
        for inew in u_det:
            det = dets[inew]
            if det.score < self.new_track:
                continue
            det.activate(self.kf, self.frame_id, self._new_id())
            activated.append(det)

        # 5. expire
        for track in self.lost:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = REMOVED
                removed.append(track)

        self.tracked = _join(_join([t for t in self.tracked if t.state == TRACKED], activated), refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, removed)
        self.tracked, self.lost = _dedupe(self.tracked, self.lost)
        self.removed = removed

        return [Track(t.track_id, tuple(float(v) for v in t.xyxy), t.score, t.det_index)
                for t in self.tracked if t.is_activated]


def _join(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _dedupe(a: list[STrack], b: list[STrack]) -> tuple[list[STrack], list[STrack]]:
    """A tracked and a lost track on the same box: keep the older one."""
    if not a or not b:
        return a, b
    dist = _iou_distance(a, b)
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(dist < 0.15)):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return [t for i, t in enumerate(a) if i not in dup_a], [t for i, t in enumerate(b) if i not in dup_b]
