import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
COLAB_DATA_ROOT = Path("/content/drive/MyDrive/Facebook Hateful Meme Dataset/data")
FRONTEND_ENV_PATH = PROJECT_ROOT / "frontend" / ".env"
POSTS_BUCKET = "post-images"


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


class PostInput(BaseModel):
    caption: str = Field(default="", description="Caption text associated with the post.")
    image_path: Optional[str] = Field(default=None, description="Readable local path to the image.")
    image_base64: Optional[str] = Field(default=None, description="Optional base64-encoded image bytes.")


class PredictionItem(BaseModel):
    harmful: bool
    label: Literal[0, 1]
    ensemble_probability: float
    multimodal_probability: float
    caption_probability: float


class BatchPredictionResponse(BaseModel):
    count: int
    multimodal_weight: float
    caption_weight: float
    ensemble_threshold: float
    items: List[PredictionItem]


class DeletePostResponse(BaseModel):
    status: Literal["deleted"]
    post_id: str


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_candidate_roots() -> List[Path]:
    candidates: List[Path] = []
    env_root = os.environ.get("HCD_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            DEFAULT_DATA_ROOT,
            PROJECT_ROOT,
            COLAB_DATA_ROOT,
        ]
    )

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key not in seen:
            unique_candidates.append(resolved)
            seen.add(key)
    return unique_candidates


def find_artifact_root() -> Path:
    required_names = [
        "caption_only_minilm_logreg.joblib",
        "mlp_best_val_model.pt",
    ]
    candidates = build_candidate_roots()
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_names):
            return candidate
    return candidates[0]


def get_allowed_origins() -> List[str]:
    env_value = os.environ.get("HCD_ALLOWED_ORIGINS")
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_config_value(*names: str) -> Optional[str]:
    frontend_env = load_env_file(FRONTEND_ENV_PATH)
    for name in names:
        value = os.environ.get(name) or frontend_env.get(name)
        if value:
            return value
    return None


class SupabaseDeleteError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SupabasePostManager:
    def __init__(self, project_url: Optional[str], publishable_key: Optional[str], bucket_name: str = POSTS_BUCKET) -> None:
        self.project_url = (project_url or "").rstrip("/")
        self.publishable_key = publishable_key or ""
        self.bucket_name = bucket_name
        self.enabled = bool(self.project_url and self.publishable_key)

    def _build_headers(self, access_token: str, accept: Optional[str] = "application/json") -> Dict[str, str]:
        headers = {
            "apikey": self.publishable_key,
            "Authorization": f"Bearer {access_token}",
        }
        if accept:
            headers["Accept"] = accept
        return headers

    def _request(
        self,
        url: str,
        method: str,
        access_token: str,
        *,
        accept: Optional[str] = "application/json",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> tuple[int, str]:
        headers = self._build_headers(access_token, accept=accept)
        if extra_headers:
            headers.update(extra_headers)

        request = urllib_request.Request(url, headers=headers, method=method)
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return response.status, payload
        except urllib_error.HTTPError as exc:
            payload = exc.read().decode("utf-8")
            detail = payload
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    detail = (
                        parsed.get("message")
                        or parsed.get("error_description")
                        or parsed.get("error")
                        or payload
                    )
            except json.JSONDecodeError:
                pass
            raise SupabaseDeleteError(exc.code, detail or f"Supabase request failed with status {exc.code}.") from exc
        except urllib_error.URLError as exc:
            raise SupabaseDeleteError(502, f"Unable to reach Supabase: {exc.reason}") from exc

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise SupabaseDeleteError(
                503,
                "Delete API is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY.",
            )

    def _fetch_post(self, post_id: str, access_token: str) -> Dict[str, Optional[str]]:
        self._require_enabled()
        query = urllib_parse.urlencode({"select": "id,image_path", "id": f"eq.{post_id}"})
        url = f"{self.project_url}/rest/v1/posts?{query}"
        try:
            _, payload = self._request(
                url,
                "GET",
                access_token,
                accept="application/vnd.pgrst.object+json",
            )
        except SupabaseDeleteError as exc:
            if exc.status_code in {404, 406}:
                raise SupabaseDeleteError(404, "Post not found or you do not have permission to delete it.") from exc
            raise
        try:
            post = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SupabaseDeleteError(502, "Supabase returned an invalid response while loading the post.") from exc
        if not isinstance(post, dict) or "id" not in post:
            raise SupabaseDeleteError(404, "Post not found or you do not have permission to delete it.")
        return post

    def _delete_storage_object(self, image_path: str, access_token: str) -> None:
        encoded_path = urllib_parse.quote(image_path, safe="/")
        url = f"{self.project_url}/storage/v1/object/{self.bucket_name}/{encoded_path}"
        try:
            self._request(url, "DELETE", access_token, accept=None)
        except SupabaseDeleteError as exc:
            if exc.status_code == 404:
                return
            raise

    def _delete_post_row(self, post_id: str, access_token: str) -> None:
        query = urllib_parse.urlencode({"id": f"eq.{post_id}"})
        url = f"{self.project_url}/rest/v1/posts?{query}"
        self._request(url, "DELETE", access_token, extra_headers={"Prefer": "return=minimal"})

    def delete_post(self, post_id: str, access_token: str) -> DeletePostResponse:
        post = self._fetch_post(post_id, access_token)
        image_path = post.get("image_path")
        if isinstance(image_path, str) and image_path:
            self._delete_storage_object(image_path, access_token)
        self._delete_post_row(post_id, access_token)
        return DeletePostResponse(status="deleted", post_id=post_id)


def extract_bearer_token(authorization_header: Optional[str]) -> str:
    if not authorization_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Authorization header must use Bearer token format.")
    return token.strip()


class HarmfulContentEnsemble:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.device = get_device()
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.multimodal_checkpoint_path = self.data_root / "mlp_best_val_model.pt"
        self.multimodal_fallback_path = self.data_root / "multimodal_fusion_head.pth"
        self.caption_model_path = self.data_root / "caption_only_minilm_logreg.joblib"
        self.caption_config_path = self.data_root / "caption_only_config.npz"
        self.ensemble_config_path = self.data_root / "ensemble_config.npz"

        self._validate_artifacts()
        self._load_models()

    def _validate_artifacts(self) -> None:
        required = [self.caption_model_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            searched_roots = "\n".join(str(path) for path in build_candidate_roots())
            raise FileNotFoundError(
                "Missing inference artifacts:\n"
                + "\n".join(missing)
                + "\nSearched roots:\n"
                + searched_roots
                + "\nRun `python prepare_local_inference_artifacts.py` to generate local artifacts."
            )
        if not self.multimodal_checkpoint_path.exists() and not self.multimodal_fallback_path.exists():
            raise FileNotFoundError(
                "Missing multimodal checkpoint. Expected mlp_best_val_model.pt or multimodal_fusion_head.pth."
            )

    def _load_models(self) -> None:
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        self.multimodal_model = AdvancedMultimodalClassifier().to(self.device)
        if self.multimodal_checkpoint_path.exists():
            checkpoint = torch.load(self.multimodal_checkpoint_path, map_location=self.device)
            self.multimodal_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            state_dict = torch.load(self.multimodal_fallback_path, map_location=self.device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.multimodal_model.load_state_dict(state_dict)
        self.multimodal_model.eval()

        caption_bundle = joblib.load(self.caption_model_path)
        self.caption_bundle_type = caption_bundle.get("bundle_type", "transformer_logreg")
        self.caption_model = caption_bundle["model"]
        self.caption_vectorizer = caption_bundle.get("vectorizer")
        self.caption_model_name = caption_bundle.get("model_name")
        self.caption_max_length = int(caption_bundle.get("max_length", 128))
        self.caption_tokenizer = None
        self.caption_encoder = None

        if self.caption_bundle_type == "transformer_logreg":
            if not self.caption_model_name:
                raise RuntimeError("Transformer caption bundle is missing `model_name`.")
            self.caption_tokenizer = AutoTokenizer.from_pretrained(self.caption_model_name)
            self.caption_encoder = AutoModel.from_pretrained(self.caption_model_name).to(self.device)
            self.caption_encoder.eval()
        elif self.caption_bundle_type != "tfidf_logreg":
            raise RuntimeError(f"Unsupported caption bundle type: {self.caption_bundle_type}")

        if self.ensemble_config_path.exists():
            ensemble_config = np.load(self.ensemble_config_path)
            self.multimodal_weight = float(ensemble_config["multimodal_weight"][0])
            self.caption_weight = float(ensemble_config["caption_weight"][0])
            self.ensemble_threshold = float(ensemble_config["ensemble_threshold"][0])
        else:
            self.multimodal_weight = 0.5
            self.caption_weight = 0.5
            self.ensemble_threshold = 0.5

    def _open_image(self, post: PostInput) -> Image.Image:
        if post.image_base64:
            image_bytes = base64.b64decode(post.image_base64)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if post.image_path:
            image_path = Path(post.image_path).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            return Image.open(image_path).convert("RGB")
        raise ValueError("Each post must provide either image_path or image_base64.")

    def _encode_caption_embeddings(self, captions: List[str], batch_size: int = 256) -> np.ndarray:
        vectors: List[np.ndarray] = []
        if self.caption_tokenizer is None or self.caption_encoder is None:
            raise RuntimeError("Transformer caption encoder is not initialized.")
        with torch.no_grad():
            for start in range(0, len(captions), batch_size):
                batch = captions[start : start + batch_size]
                inputs = self.caption_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.caption_max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.caption_encoder(**inputs)
                pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(vectors)

    def _predict_caption_probabilities(self, captions: List[str]) -> np.ndarray:
        if self.caption_bundle_type == "tfidf_logreg":
            if self.caption_vectorizer is None:
                raise RuntimeError("TF-IDF caption bundle is missing `vectorizer`.")
            caption_features = self.caption_vectorizer.transform(captions)
            return self.caption_model.predict_proba(caption_features)[:, 1].astype(np.float32, copy=False)

        caption_embeddings = self._encode_caption_embeddings(captions)
        return self.caption_model.predict_proba(np.asarray(caption_embeddings))[:, 1].astype(np.float32, copy=False)

    def _encode_multimodal_embeddings(
        self, images: List[Image.Image], captions: List[str], batch_size: int = 64
    ) -> tuple[np.ndarray, np.ndarray]:
        image_vectors: List[np.ndarray] = []
        text_vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(captions), batch_size):
                batch_images = images[start : start + batch_size]
                batch_captions = captions[start : start + batch_size]
                inputs = self.clip_processor(
                    text=batch_captions,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.clip_model(**inputs, return_dict=True)
                batch_image_features = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
                batch_text_features = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
                image_vectors.append(batch_image_features.cpu().numpy().astype(np.float32, copy=False))
                text_vectors.append(batch_text_features.cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(image_vectors), np.vstack(text_vectors)

    def predict(self, posts: List[PostInput]) -> BatchPredictionResponse:
        if not posts:
            raise ValueError("At least one post is required.")

        images = [self._open_image(post) for post in posts]
        captions = [post.caption for post in posts]

        caption_probabilities = self._predict_caption_probabilities(captions)

        image_features, text_features = self._encode_multimodal_embeddings(images, captions)
        with torch.no_grad():
            image_tensor = torch.from_numpy(image_features).to(device=self.device, dtype=torch.float32)
            text_tensor = torch.from_numpy(text_features).to(device=self.device, dtype=torch.float32)
            multimodal_logits = self.multimodal_model(image_tensor, text_tensor)
            multimodal_probabilities = torch.sigmoid(multimodal_logits).squeeze(1).cpu().numpy().astype(
                np.float32, copy=False
            )

        ensemble_probabilities = (
            self.multimodal_weight * multimodal_probabilities + self.caption_weight * caption_probabilities
        ).astype(np.float32, copy=False)
        labels = (ensemble_probabilities >= self.ensemble_threshold).astype(np.int64)

        items = [
            PredictionItem(
                harmful=bool(label),
                label=int(label),
                ensemble_probability=float(ensemble_probability),
                multimodal_probability=float(multimodal_probability),
                caption_probability=float(caption_probability),
            )
            for label, ensemble_probability, multimodal_probability, caption_probability in zip(
                labels,
                ensemble_probabilities,
                multimodal_probabilities,
                caption_probabilities,
            )
        ]

        return BatchPredictionResponse(
            count=len(items),
            multimodal_weight=self.multimodal_weight,
            caption_weight=self.caption_weight,
            ensemble_threshold=self.ensemble_threshold,
            items=items,
        )


def create_app() -> FastAPI:
    data_root = find_artifact_root()
    post_manager = SupabasePostManager(
        project_url=get_config_value("HCD_SUPABASE_URL", "VITE_SUPABASE_URL"),
        publishable_key=get_config_value(
            "HCD_SUPABASE_PUBLISHABLE_KEY",
            "VITE_SUPABASE_PUBLISHABLE_KEY",
            "VITE_SUPABASE_ANON_KEY",
        ),
    )
    try:
        engine = HarmfulContentEnsemble(data_root)
    except Exception as exc:  # pragma: no cover - startup error path
        raise RuntimeError(f"Failed to initialize inference engine: {exc}") from exc

    app = FastAPI(title="Harmful Content Detection API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(posts: List[PostInput]) -> BatchPredictionResponse:
        try:
            return engine.predict(posts)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - API error path
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/posts/{post_id}", response_model=DeletePostResponse)
    def delete_post(post_id: str, authorization: Optional[str] = Header(default=None)) -> DeletePostResponse:
        access_token = extract_bearer_token(authorization)
        try:
            return post_manager.delete_post(post_id, access_token)
        except SupabaseDeleteError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return app


app = create_app()
