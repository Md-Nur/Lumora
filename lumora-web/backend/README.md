---
title: Lumora
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

Lumora FastAPI inference Space for chest X-ray and CT report generation.

Endpoints:

- `GET /`
- `POST /predict` for X-ray `.jpg`, `.jpeg`, `.png`
- `POST /predict/ct` for CT `.nii`, `.nii.gz`, `.dcm`

Local run:

```bash
cd lumora-web/backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 7860
```

The backend first uses `XRAY_MODEL_PATH` / `CT_MODEL_PATH` when set, then nearby
local checkpoint files, and finally downloads from the configured Hugging Face
model repos.
