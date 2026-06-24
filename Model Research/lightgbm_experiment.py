import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np


DEFAULT_EMBEDDING_ROOT = Path("/Users/debrajgoswami/Desktop/Harmful Content Detection/clip_embeddings")


def load_split_arrays(embedding_root: Path, split_name: str):
    image_path = embedding_root / f"{split_name}_images.npy"
    text_path = embedding_root / f"{split_name}_text.npy"
    label_path = embedding_root / f"{split_name}_labels.npy"

    missing = [str(path) for path in (image_path, text_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing embedding files:\n" + "\n".join(missing))

    images = np.load(image_path, mmap_mode="r")
    text = np.load(text_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")

    if len(images) != len(text) or len(images) != len(labels):
        raise ValueError(f"Split {split_name} has mismatched row counts across image/text/label arrays.")

    return images, text, labels


def build_hadamard_fused_matrix(images: np.ndarray, text: np.ndarray) -> np.ndarray:
    interaction = images * text
    return np.hstack([images, text, interaction]).astype(np.float32, copy=False)


def prepare_split(embedding_root: Path, split_name: str):
    images, text, labels = load_split_arrays(embedding_root, split_name)
    fused = build_hadamard_fused_matrix(images, text)
    return fused, np.asarray(labels, dtype=np.int32)


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    num_negative = int((labels == 0).sum())
    num_positive = int((labels == 1).sum())

    if num_positive == 0:
        raise ValueError("Training labels contain no positive examples; cannot compute scale_pos_weight.")

    return num_negative / num_positive


def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM classifier on fused CLIP embeddings.")
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Directory containing train/val image, text, and label .npy files.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=200,
        help="Number of LightGBM boosting rounds.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("/Users/debrajgoswami/Desktop/Harmful Content Detection/lightgbm_model.txt"),
        help="Where to save the trained LightGBM model.",
    )
    args = parser.parse_args()

    embedding_root = args.embedding_root.expanduser().resolve()
    print(f"Using embedding root: {embedding_root}")

    X_train, y_train = prepare_split(embedding_root, "train")
    X_val, y_val = prepare_split(embedding_root, "val")

    print(f"Train fused shape: {X_train.shape}")
    print(f"Validation fused shape: {X_val.shape}")

    if X_train.shape[1] != 1536 or X_val.shape[1] != 1536:
        raise ValueError(
            f"Expected 1536 fused features, got train={X_train.shape[1]} and val={X_val.shape[1]}."
        )

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")

    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 1,
        "device_type": "cpu",
    }

    booster = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.output_model))
    print(f"Saved model to: {args.output_model}")


if __name__ == "__main__":
    main()
