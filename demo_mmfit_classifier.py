import argparse
import csv
import json
import os
from collections import Counter
from typing import List

import joblib
import numpy as np

from mmfit_classifier_utils import (
    TARGET_ACTIONS,
    extract_segment_coords,
    fill_missing,
    load_pose_2d,
    normalize_coords,
    window_features,
)


def load_worker_segments(labels_csv: str, actions: List[str]):
    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            start, end, reps, action = row[:4]
            if action not in actions:
                continue
            rows.append(
                {
                    "start": int(float(start)),
                    "end": int(float(end)),
                    "reps": int(float(reps)),
                    "action": action,
                }
            )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demo inference for MM-Fit 4-action classifier.")
    p.add_argument("--dataset_root", default="mm-fit", help="Path to MM-Fit root folder.")
    p.add_argument(
        "--model_path",
        default="artifacts/mmfit_action_cls/rf_mmfit_4actions.joblib",
        help="Path to trained model.",
    )
    p.add_argument(
        "--meta_path",
        default="artifacts/mmfit_action_cls/meta.json",
        help="Path to metadata json from training.",
    )
    p.add_argument("--worker", default="w00", help="Worker id, e.g. w00.")
    p.add_argument("--segment_index", type=int, default=0, help="Segment index in filtered action list.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    clf = joblib.load(args.model_path)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    actions = meta.get("actions", list(TARGET_ACTIONS))
    window_size = int(meta["window_size"])
    stride = int(meta["stride"])

    worker_dir = os.path.join(args.dataset_root, args.worker)
    labels_csv = os.path.join(worker_dir, f"{args.worker}_labels.csv")
    segments = load_worker_segments(labels_csv, actions)
    if not segments:
        raise RuntimeError(f"No matching segments for worker={args.worker} with actions={actions}")
    if args.segment_index < 0 or args.segment_index >= len(segments):
        raise IndexError(f"segment_index out of range: 0..{len(segments)-1}")

    seg = segments[args.segment_index]
    pose = load_pose_2d(worker_dir, args.worker)
    coords = extract_segment_coords(pose, seg["start"], seg["end"])
    coords = fill_missing(coords)
    coords = normalize_coords(coords)
    feats = window_features(coords, window_size=window_size, stride=stride)
    if feats.shape[0] == 0:
        raise RuntimeError("Segment too short for current window_size.")

    pred_ids = clf.predict(feats)
    prob = clf.predict_proba(feats)
    mean_prob = prob.mean(axis=0)

    vote = Counter(pred_ids.tolist()).most_common(1)[0][0]
    pred_label = actions[vote]
    top3_idx = np.argsort(-mean_prob)[:3]

    print("Demo segment:")
    print(f"worker={args.worker}, segment_index={args.segment_index}")
    print(f"gt_action={seg['action']}, reps={seg['reps']}, frame=[{seg['start']},{seg['end']}]")
    print(f"windows={len(pred_ids)}, predicted_action={pred_label}")
    print("Top-3 average probabilities:")
    for i in top3_idx:
        print(f"  {actions[int(i)]}: {mean_prob[int(i)]:.4f}")


if __name__ == "__main__":
    main()
