# Backend Export

This folder contains the FastAPI inference backend prepared for Render deployment.

## Render

- Root directory: `project_exports/Backend` if deploying from the repo root, or this folder directly if uploaded separately
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

### Render Environment Variables

- `HCD_ALLOWED_ORIGINS=https://your-netlify-site.netlify.app`
- `HCD_DATA_ROOT=./data`
- `HCD_SUPABASE_URL=https://your-project.supabase.co`
- `HCD_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key`

## Included

- `fastapi_backend.py`: API entrypoint
- `requirements.txt`: Python dependencies
- `data/`: model artifacts required for inference
- `render.yaml`: starter Render service definition

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000
```
