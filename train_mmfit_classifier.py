import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from mmfit_classifier_utils import (
    TARGET_ACTIONS,
    build_dataset,
    mask_by_workers,
    per_class_counts,
    split_workers,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MM-Fit 4-action classifier.")
    p.add_argument("--dataset_root", default="mm-fit", help="Path to MM-Fit root folder.")
    p.add_argument("--window_size", type=int, default=64, help="Window length in frames.")
    p.add_argument("--stride", type=int, default=32, help="Window stride in frames.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output_dir", default="artifacts/mmfit_action_cls", help="Directory to save model and metadata."
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    actions = TARGET_ACTIONS

    X, y, workers, action_to_id, label_names = build_dataset(
        dataset_root=args.dataset_root,
        actions=actions,
        window_size=args.window_size,
        stride=args.stride,
    )
    split = split_workers(workers, seed=args.seed)

    train_mask = mask_by_workers(workers, split["train"])
    val_mask = mask_by_workers(workers, split["val"])
    test_mask = mask_by_workers(workers, split["test"])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=args.seed,
        n_jobs=1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    def evaluate(name: str, x: np.ndarray, yy: np.ndarray) -> None:
        pred = clf.predict(x)
        acc = accuracy_score(yy, pred)
        print(f"\n[{name}] samples={len(yy)} accuracy={acc:.4f}")
        print(classification_report(yy, pred, target_names=label_names, digits=4))

    print("Workers split:")
    print("train:", split["train"])
    print("val  :", split["val"])
    print("test :", split["test"])
    print("\nTrain class counts:", per_class_counts(y_train, label_names))
    print("Val class counts  :", per_class_counts(y_val, label_names))
    print("Test class counts :", per_class_counts(y_test, label_names))

    evaluate("VAL", X_val, y_val)
    evaluate("TEST", X_test, y_test)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "rf_mmfit_4actions.joblib")
    meta_path = os.path.join(args.output_dir, "meta.json")

    joblib.dump(clf, model_path)
    meta = {
        "actions": label_names,
        "action_to_id": action_to_id,
        "window_size": args.window_size,
        "stride": args.stride,
        "split": split,
        "feature_dim": int(X.shape[1]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"\n[OK] model saved: {model_path}")
    print(f"[OK] meta saved : {meta_path}")


if __name__ == "__main__":
    main()
