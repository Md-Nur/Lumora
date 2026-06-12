Lumora backend now uses a single FastAPI entrypoint in the Hugging Face Space
checkout:

```bash
cd lumora-web/backend/lumora
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 7860
```

The same `main.py` is used for local development and for the Hugging Face Space.
