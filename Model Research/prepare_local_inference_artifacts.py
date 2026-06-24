import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
EMBEDDING_ROOT = PROJECT_ROOT / "clip_embeddings"


class AdvancedMultimodalClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.image_projection = nn.Linear(embedding_dim, hidden_dim)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim)
        self.image_to_text_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_image_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_image = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mish = nn.Mish()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_token = self.image_projection(image_features).unsqueeze(1)
        text_token = self.text_projection(text_features).unsqueeze(1)

        attended_image, _ = self.image_to_text_attention(image_token, text_token, text_token)
        attended_text, _ = self.text_to_image_attention(text_token, image_token, image_token)

        image_token = self.mish(self.norm_image(image_token + self.dropout(attended_image)))
        text_token = self.mish(self.norm_text(text_token + self.dropout(attended_text)))

        fused = torch.cat([image_token.squeeze(1), text_token.squeeze(1)], dim=-1)
        return self.fusion_mlp(fused)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl_records(file_path: Path) -> List[Dict]:
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_caption_split(split_name: str) -> Tuple[List[str], np.ndarray]:
    file_name = "dev.jsonl" if split_name == "val" else f"{split_name}.jsonl"
    records = read_jsonl_records(DATA_ROOT / file_name)
    captions = [str(record.get("text", "")) for record in records]
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    return captions, labels


def train_caption_model() -> Tuple[np.ndarray, np.ndarray]:
    train_captions, train_labels = load_caption_split("train")
    val_captions, val_labels = load_caption_split("val")

    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_features=50000)
    x_train = vectorizer.fit_transform(train_captions)
    x_val = vectorizer.transform(val_captions)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, train_labels)

    val_probabilities = model.predict_proba(x_val)[:, 1].astype(np.float32, copy=False)
    val_predictions = (val_probabilities >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels,
        val_predictions,
        average="binary",
        zero_division=0,
    )
    print(
        f"Caption model validation at 0.50 threshold -> "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )

    joblib.dump(
        {
            "bundle_type": "tfidf_logreg",
            "vectorizer": vectorizer,
            "model": model,
        },
        DATA_ROOT / "caption_only_minilm_logreg.joblib",
    )
    np.savez(
        DATA_ROOT / "caption_only_config.npz",
        default_threshold=np.asarray([0.5], dtype=np.float32),
    )
    return val_labels.astype(np.float32, copy=False), val_probabilities


def load_embedding_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_features = np.load(EMBEDDING_ROOT / f"{split_name}_images.npy").astype(np.float32, copy=False)
    text_features = np.load(EMBEDDING_ROOT / f"{split_name}_text.npy").astype(np.float32, copy=False)
    labels = np.load(EMBEDDING_ROOT / f"{split_name}_labels.npy").astype(np.float32, copy=False)
    return image_features, text_features, labels


def evaluate_multimodal(
    model: AdvancedMultimodalClassifier,
    loader: DataLoader,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for image_batch, text_batch, label_batch in loader:
            image_batch = image_batch.to(device=device, dtype=torch.float32)
            text_batch = text_batch.to(device=device, dtype=torch.float32)
            logits = model(image_batch, text_batch)
            batch_probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            probabilities.append(batch_probabilities.astype(np.float32, copy=False))
            targets.append(label_batch.numpy().astype(np.float32, copy=False))
    all_probabilities = np.concatenate(probabilities)
    all_targets = np.concatenate(targets)
    score = f1_score(all_targets.astype(np.int64), (all_probabilities >= 0.5).astype(np.int64), zero_division=0)
    return score, all_targets, all_probabilities


def train_multimodal_model() -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()
    train_images, train_text, train_labels = load_embedding_split("train")
    val_images, val_text, val_labels = load_embedding_split("val")

    train_dataset = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_text),
        torch.from_numpy(train_labels),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_images),
        torch.from_numpy(val_text),
        torch.from_numpy(val_labels),
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = AdvancedMultimodalClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))

    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    best_probabilities = None

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for image_batch, text_batch, label_batch in train_loader:
            image_batch = image_batch.to(device=device, dtype=torch.float32)
            text_batch = text_batch.to(device=device, dtype=torch.float32)
            label_batch = label_batch.to(device=device, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(image_batch, text_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size = image_batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        val_f1, _, val_probabilities = evaluate_multimodal(model, val_loader, device)
        avg_train_loss = total_loss / max(total_examples, 1)
        print(f"Epoch {epoch:02d} -> train_loss={avg_train_loss:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_probabilities = val_probabilities
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if not best_state:
        raise RuntimeError("Failed to train multimodal model.")

    checkpoint = {
        "model_state_dict": best_state,
        "best_val_f1": best_f1,
        "device_used_for_training": device,
    }
    torch.save(checkpoint, DATA_ROOT / "mlp_best_val_model.pt")
    torch.save(best_state, DATA_ROOT / "multimodal_fusion_head.pth")
    return val_labels.astype(np.float32, copy=False), best_probabilities.astype(np.float32, copy=False)


def tune_ensemble(
    targets: np.ndarray,
    multimodal_probabilities: np.ndarray,
    caption_probabilities: np.ndarray,
) -> Tuple[float, float, float, float]:
    best = (-1.0, 0.5, 0.5, 0.5)
    for multimodal_weight in np.linspace(0.0, 1.0, 21):
        caption_weight = 1.0 - float(multimodal_weight)
        blended = multimodal_weight * multimodal_probabilities + caption_weight * caption_probabilities
        for threshold in np.linspace(0.1, 0.9, 33):
            predictions = (blended >= threshold).astype(np.int64)
            f1 = f1_score(targets.astype(np.int64), predictions, zero_division=0)
            if f1 > best[0]:
                best = (f1, float(multimodal_weight), float(caption_weight), float(threshold))
    return best


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    caption_targets, caption_probabilities = train_caption_model()
    multimodal_targets, multimodal_probabilities = train_multimodal_model()

    if not np.array_equal(caption_targets.astype(np.int64), multimodal_targets.astype(np.int64)):
        raise RuntimeError("Caption and multimodal validation targets do not align.")

    best_f1, multimodal_weight, caption_weight, threshold = tune_ensemble(
        caption_targets,
        multimodal_probabilities,
        caption_probabilities,
    )
    np.savez(
        DATA_ROOT / "ensemble_config.npz",
        multimodal_weight=np.asarray([multimodal_weight], dtype=np.float32),
        caption_weight=np.asarray([caption_weight], dtype=np.float32),
        ensemble_threshold=np.asarray([threshold], dtype=np.float32),
        best_val_f1=np.asarray([best_f1], dtype=np.float32),
    )
    print(
        f"Saved inference artifacts to {DATA_ROOT} "
        f"(multimodal_weight={multimodal_weight:.2f}, caption_weight={caption_weight:.2f}, threshold={threshold:.2f}, "
        f"best_val_f1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()
