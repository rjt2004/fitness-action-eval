import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


TARGET_ACTIONS = ("squats", "pushups", "lunges", "situps")


@dataclass
class Segment:
    worker: str
    start_frame: int
    end_frame: int
    reps: int
    action: str


def load_labels(labels_csv: str, target_actions: Sequence[str]) -> List[Segment]:
    segments: List[Segment] = []
    keep = set(target_actions)
    worker = os.path.basename(labels_csv).split("_")[0]
    with open(labels_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            start, end, reps, action = row[:4]
            if action not in keep:
                continue
            segments.append(
                Segment(
                    worker=worker,
                    start_frame=int(float(start)),
                    end_frame=int(float(end)),
                    reps=int(float(reps)),
                    action=action,
                )
            )
    return segments


def load_pose_2d(worker_dir: str, worker: str) -> np.ndarray:
    pose_path = os.path.join(worker_dir, f"{worker}_pose_2d.npy")
    pose = np.load(pose_path)
    # MM-Fit pose_2d shape: (2, T, 19), joint index 0 is frame id marker.
    return pose


def frame_to_index(pose: np.ndarray, frame_number: int) -> int:
    first_frame = int(round(float(pose[0, 0, 0])))
    return frame_number - first_frame


def extract_segment_coords(pose: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    start_idx = frame_to_index(pose, start_frame)
    end_idx = frame_to_index(pose, end_frame)
    start_idx = max(0, start_idx)
    end_idx = min(pose.shape[1] - 1, end_idx)
    if end_idx <= start_idx:
        return np.empty((0, 18, 2), dtype=np.float32)

    # Use joints 1..18, dropping index 0 metadata channel.
    x = pose[0, start_idx : end_idx + 1, 1:19]
    y = pose[1, start_idx : end_idx + 1, 1:19]
    coords = np.stack([x, y], axis=-1).astype(np.float32)  # (T, 18, 2)

    # MM-Fit missing sentinel is 4050.
    coords[coords >= 4000] = np.nan
    return coords


def _fill_nan_1d(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    idx = np.arange(n)
    good = np.isfinite(arr)
    if good.sum() == 0:
        return np.zeros_like(arr)
    if good.sum() == 1:
        arr[~good] = arr[good][0]
        return arr
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return arr


def fill_missing(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    if out.size == 0:
        return out
    t, j, c = out.shape
    for joint in range(j):
        for ch in range(c):
            out[:, joint, ch] = _fill_nan_1d(out[:, joint, ch])
    return out


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords
    # Per-frame center/scale normalization for viewpoint robustness.
    center = np.nanmean(coords, axis=1, keepdims=True)  # (T,1,2)
    centered = coords - center
    scale = np.nanstd(centered.reshape(centered.shape[0], -1), axis=1, keepdims=True) + 1e-6
    scale = scale[:, None, :]
    return centered / scale


def window_features(coords: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    feats = []
    t = coords.shape[0]
    if t < window_size:
        return np.empty((0, 1), dtype=np.float32)
    for st in range(0, t - window_size + 1, stride):
        win = coords[st : st + window_size]  # (W, 18, 2)
        flat = win.reshape(window_size, -1)  # (W, 36)
        vel = np.diff(flat, axis=0)
        feat = np.concatenate(
            [
                np.mean(flat, axis=0),
                np.std(flat, axis=0),
                np.min(flat, axis=0),
                np.max(flat, axis=0),
                np.mean(vel, axis=0),
                np.std(vel, axis=0),
            ],
            axis=0,
        )
        feats.append(feat.astype(np.float32))
    return np.asarray(feats, dtype=np.float32)


def build_dataset(
    dataset_root: str,
    actions: Sequence[str] = TARGET_ACTIONS,
    window_size: int = 64,
    stride: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    action_to_id = {a: i for i, a in enumerate(actions)}
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    workers: List[str] = []

    worker_dirs = sorted(
        [d for d in os.listdir(dataset_root) if d.startswith("w") and os.path.isdir(os.path.join(dataset_root, d))]
    )
    for worker in worker_dirs:
        worker_dir = os.path.join(dataset_root, worker)
        labels_csv = os.path.join(worker_dir, f"{worker}_labels.csv")
        if not os.path.exists(labels_csv):
            continue
        segments = load_labels(labels_csv, actions)
        if not segments:
            continue
        pose = load_pose_2d(worker_dir, worker)
        for seg in segments:
            coords = extract_segment_coords(pose, seg.start_frame, seg.end_frame)
            if coords.shape[0] == 0:
                continue
            coords = fill_missing(coords)
            coords = normalize_coords(coords)
            feats = window_features(coords, window_size=window_size, stride=stride)
            if feats.shape[0] == 0:
                continue
            X_list.append(feats)
            y_list.append(np.full((feats.shape[0],), action_to_id[seg.action], dtype=np.int64))
            workers.extend([worker] * feats.shape[0])

    if not X_list:
        raise RuntimeError("No training samples generated. Check dataset path and window settings.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    w = np.asarray(workers)
    return X, y, w, action_to_id, list(actions)


def split_workers(
    workers: Sequence[str], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> Dict[str, List[str]]:
    uniq = np.array(sorted(set(workers)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    train = uniq[:n_train].tolist()
    val = uniq[n_train : n_train + n_val].tolist()
    test = uniq[n_train + n_val :].tolist()
    return {"train": train, "val": val, "test": test}


def mask_by_workers(worker_arr: np.ndarray, selected: Sequence[str]) -> np.ndarray:
    selected_set = set(selected)
    return np.asarray([w in selected_set for w in worker_arr], dtype=bool)


def per_class_counts(y: np.ndarray, label_names: Sequence[str]) -> Dict[str, int]:
    counts = defaultdict(int)
    for label in y:
        counts[label_names[int(label)]] += 1
    return dict(counts)
