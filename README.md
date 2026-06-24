# Seive: Multimodal Harmful Content Detection for Social Media

## Abstract

Seive is a research-driven social media moderation prototype that studies harmful content detection at the level of an individual post composed of an image and a caption. The project investigates whether combining a caption-only classifier with a multimodal image-text classifier yields a stronger moderation signal than either branch alone. The result is a deployable ensemble-based inference system integrated into a full-stack application built with React, FastAPI, and Supabase.

This repository is structured as a portfolio artifact rather than only an application codebase. It documents the modeling rationale, training pipeline, exported inference artifacts, and product integration pathway that turns research experiments into a deployable moderation workflow.

## Project Framing

Online harmful content moderation is fundamentally multimodal. A caption may appear benign in isolation while its paired image changes the semantic intent of the post. Similarly, an image may be ambiguous until caption context makes its harmful meaning explicit. For that reason, Seive treats moderation as a binary multimodal classification problem over:

- image content
- caption text

Instead of hard-blocking content, the product stores posts and attaches model-derived metadata:

- `model_label`
- branch-specific probabilities
- ensemble probability
- user safety feedback

This design makes the system appropriate for research, auditing, and explainable portfolio demonstration, while still being deployable as a functioning social application.

## Portfolio Highlights

- Designed and trained a multimodal harmful-content detection pipeline
- Combined classical NLP and neural multimodal modeling into a validation-tuned ensemble
- Exported inference artifacts for lightweight deployment
- Integrated the moderation model into a production-style social app workflow
- Built a deployable frontend, backend, and research package split for cloud hosting and GitHub presentation

## Repository Layout

```text
project_exports/
  Frontend/
  Backend/
  Model Research/
```

- `Frontend/` contains the React + Vite client prepared for Netlify
- `Backend/` contains the FastAPI inference service prepared for Render
- `Model Research/` contains notebooks, annotations, research scripts, and training artifacts

## Research Objective

The central research question of Seive is:

> Can a lightweight ensemble of text-only and multimodal classifiers provide a practical, interpretable, and deployable harmful-content detection system for social media posts?

The current system answers this through three components:

1. a caption-only branch for textual harm signals
2. a multimodal branch for image-text interaction modeling
3. a tuned ensemble that blends both outputs and learns an operating threshold on validation data

## Methodology

### Problem Formulation

Each example is treated as a binary labeled post:

- `0`: safe
- `1`: harmful

The input consists of:

- caption text from JSONL annotation files
- paired image-text embeddings for multimodal experiments

The project emphasizes practical moderation inference rather than pure benchmark optimization. The modeling decisions therefore prioritize:

- modularity
- explainability
- reproducibility
- ease of deployment

### Data Assets

The research export includes the core labeled splits:

- `Model Research/data/train.jsonl`
- `Model Research/data/dev.jsonl`
- `Model Research/data/test.jsonl`

These files provide the textual labels used in training and validation. During experimentation, the project also used precomputed CLIP-aligned image and text embedding arrays. Those large intermediate files are intentionally excluded from this GitHub-oriented export to keep the repository lighter, but the downstream training and inference code still reflects their role in the methodology.

### Representation Learning

The multimodal branch relies on aligned CLIP feature spaces. During research, image and caption representations were extracted into split-specific arrays, allowing rapid iteration on fusion architectures without repeatedly recomputing expensive embeddings.

At deployment time, the backend computes embeddings on demand using:

- `openai/clip-vit-base-patch32`

This separation is deliberate:

- research uses cached representations for faster experimentation
- deployed inference recomputes features live so the service remains self-contained

### Caption-Only Model

The caption-only branch is trained in `Model Research/prepare_local_inference_artifacts.py`. The current implementation uses a classical lexical baseline:

- `TfidfVectorizer`
- `LogisticRegression`

The training configuration includes:

- lowercase normalization
- Unicode accent stripping
- unigram and bigram features
- `min_df=2`
- `max_features=50000`
- `class_weight="balanced"`
- `liblinear` optimization

This branch serves several purposes:

- captures explicit textual abuse and threat patterns
- provides a strong and interpretable baseline
- remains computationally cheap at inference time
- complements multimodal reasoning when image evidence is weak

The exported artifacts are:

- `caption_only_minilm_logreg.joblib`
- `caption_only_config.npz`

Although the file name suggests a transformer-based caption model, the current exported artifact format also supports TF-IDF + logistic regression bundles. The serving code is intentionally flexible enough to support future transformer-backed caption classifiers without changing the API contract.

### Multimodal Model

The multimodal classifier is also trained in `Model Research/prepare_local_inference_artifacts.py`. It consumes paired image and text embeddings and passes them through `AdvancedMultimodalClassifier`, a compact fusion architecture composed of:

- learned image projection
- learned text projection
- bidirectional cross-attention
- residual updates with layer normalization
- Mish nonlinearities
- dropout regularization
- a fusion MLP classification head

The architectural intuition is that harmful content frequently emerges from cross-modal interaction rather than from either modality independently. The model therefore allows image features to condition text features and vice versa before computing the final harmfulness score.

The training loop:

- loads train and validation embedding splits
- uses `AdamW`
- trains for 10 epochs
- compensates for class imbalance through `BCEWithLogitsLoss(pos_weight=...)`
- keeps the checkpoint with the best validation F1

The main exported multimodal artifacts are:

- `mlp_best_val_model.pt`
- `multimodal_fusion_head.pth`

### Ensemble Construction

The deployed moderation signal is not produced by a single model. Instead, the project uses a weighted ensemble over the caption-only and multimodal probabilities.

After both branches are trained, `prepare_local_inference_artifacts.py`:

1. computes validation probabilities for each branch
2. sweeps multimodal-caption weight combinations
3. sweeps decision thresholds
4. selects the configuration with the best validation F1

The final ensemble configuration is stored in:

- `ensemble_config.npz`

This artifact contains:

- `multimodal_weight`
- `caption_weight`
- `ensemble_threshold`
- `best_val_f1`

At inference time, the final moderation score is:

```text
ensemble_probability =
  multimodal_weight * multimodal_probability +
  caption_weight * caption_probability
```

The binary harmful label is assigned when:

```text
ensemble_probability >= ensemble_threshold
```

This approach was chosen because it is:

- simple to reason about
- easy to debug
- straightforward to recalibrate
- well aligned with production deployment constraints

### Additional Baseline: LightGBM Fusion

The file `Model Research/lightgbm_experiment.py` explores an alternative family of models. It concatenates:

- CLIP image embeddings
- CLIP text embeddings
- elementwise Hadamard interaction features

and trains a LightGBM classifier on the resulting fused representation.

This experiment is useful for portfolio depth because it shows that the project did not rely on a single modeling assumption. Instead, it compares neural fusion against a tree-based baseline over handcrafted multimodal feature composition.

## Inference System

The deployable moderation service is implemented in `Backend/fastapi_backend.py`. At startup, the service:

1. locates the artifact directory
2. validates the presence of required caption and multimodal files
3. loads the caption classifier
4. loads the multimodal checkpoint
5. loads the ensemble weights and threshold
6. loads CLIP for live feature extraction

For an incoming post, the backend performs the following sequence:

1. decode the image from file path or base64
2. read the caption
3. compute caption probability
4. compute CLIP image embedding
5. compute CLIP text embedding
6. run the multimodal fusion network
7. combine branch outputs with the stored ensemble weights
8. threshold the final score
9. return branch and ensemble probabilities

The main inference endpoint is:

- `POST /predict-batch`

Each prediction item returns:

- `harmful`
- `label`
- `ensemble_probability`
- `multimodal_probability`
- `caption_probability`

This output format is intentionally transparent. It allows downstream UI and evaluation workflows to inspect both the final decision and the contribution of individual model branches.

## Product Integration

### Frontend

The frontend in `Frontend/` is a React + Vite application responsible for:

- Supabase authentication
- feed rendering
- post creation with image upload
- moderation call before persistence
- display of model labeling
- safe/unsafe community feedback
- deletion of owned posts

The application treats the model as an assistive moderation layer rather than a hard gate. This is a deliberate design choice that makes the product more suitable for experimentation and analysis of model behavior in a social setting.

### Backend

The backend in `Backend/` is a FastAPI service that exposes:

- `GET /health`
- `POST /predict-batch`
- `DELETE /posts/{post_id}`

In addition to inference, it handles:

- artifact loading
- CORS configuration
- lightweight request validation
- authenticated delete flow through Supabase

### Data Layer

Supabase is used for:

- authentication
- post metadata
- image storage
- harmful reaction records

The schema preserves not only user-generated content but also model outputs:

- `model_label`
- `model_ensemble_probability`
- `model_multimodal_probability`
- `model_caption_probability`

This creates a clean bridge between machine learning inference and application state.

## Design Rationale

The project deliberately avoids an oversized end-to-end fine-tuned moderation stack. Instead, it chooses a modular design that is easier to iterate on, explain, and deploy. The rationale is as follows:

- text-only models are inexpensive and surprisingly strong on explicit harmful language
- multimodal models are necessary when harm is only visible through image-text interaction
- ensembles often outperform individual branches by compensating for different failure modes
- artifact-based deployment is easier to serve reliably than a monolithic research training stack

This makes Seive a useful portfolio piece because it demonstrates both modeling judgment and engineering pragmatism.

## Limitations

The current system should be understood as a prototype and research demonstration. Important limitations remain:

- no OCR signal is currently included
- the multimodal branch uses fixed CLIP representations rather than end-to-end fine-tuning
- calibration could be improved further
- the GitHub export omits some large intermediate files used during experimentation
- moderation remains binary even though real-world harm is more nuanced

## Future Work

The roadmap in `Model Research/improvements.txt` identifies several next steps:

- add OCR extraction and use embedded text as an additional modality
- replace frozen-feature training with stronger end-to-end vision-language fine-tuning
- expand cross-modal fusion strategies
- introduce a higher-precision two-stage moderation pipeline
- ensemble neural, tree-based, and text-only systems
- recalibrate probabilities on validation data

These directions would make the system more realistic, more robust, and more aligned with real moderation settings.

## Included Research Assets

The research export includes:

- `index.ipynb`
- `test.ipynb`
- `data/train.jsonl`
- `data/dev.jsonl`
- `data/test.jsonl`
- `data/caption_only_minilm_logreg.joblib`
- `data/caption_only_config.npz`
- `data/mlp_best_val_model.pt`
- `data/multimodal_fusion_head.pth`
- `data/ensemble_config.npz`
- `lightgbm_experiment.py`
- `prepare_local_inference_artifacts.py`
- `improvements.txt`

To keep the repository smaller, the export intentionally excludes:

- `clip_embeddings/`
- `test/`

These omitted directories are not required for deployed inference, but they may be useful for reproducing every research step in full.

## Deployment Context

- `Frontend/` is prepared for Netlify
- `Backend/` is prepared for Render
- `Model Research/` is prepared for GitHub documentation and research presentation

Together, these folders represent the full lifecycle of the project:

1. labeled data and experimentation
2. artifact generation
3. inference deployment
4. product integration

## Summary

Seive is best understood as a full-stack machine learning portfolio project centered on multimodal harmful content detection. Its main contribution is not just a trained classifier, but a coherent end-to-end path from research experimentation to deployable inference and product integration.

The project demonstrates:

- multimodal ML system design
- practical ensemble construction
- artifact-based inference deployment
- frontend-backend integration for applied AI products
