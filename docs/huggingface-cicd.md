# Hugging Face CI/CD

Lumora deploys the FastAPI inference backend to a Hugging Face Docker Space and keeps model checkpoints in separate Hugging Face model repositories.

## GitHub Secrets And Variables

Add this GitHub Actions secret:

- `HF_TOKEN`: Hugging Face write token. Prefer a fine-grained token scoped to the target Space and model repos.

Add this GitHub Actions repository variable:

- `HF_SPACE_ID`: target Space repo id, for example `nur9211/lumora`.

## Hugging Face Space Variables

Set these in the Space settings:

- `HF_TOKEN`: read token if either model repo is private.
- `XRAY_MODEL_REPO`: defaults to `nur9211/mimic-vlm-model`.
- `XRAY_MODEL_FILENAME`: defaults to `mimic_vlm_phase2_fully_trained.pt`.
- `CT_MODEL_REPO`: defaults to `nur9211/ct-rate-vlm-model`.
- `CT_MODEL_FILENAME`: defaults to `ct_rate_vlm_phase2_fully_trained.pt`.
- `CORS_ALLOW_ORIGINS`: comma-separated frontend origins.

## Upload The CT Model From This Machine

Run this after logging in with `huggingface-cli login`, or pass `--token`.

```bash
uv run python scripts/upload_hf_model.py \
  --repo-id nur9211/ct-rate-vlm-model \
  --checkpoint checkpoints/ct_rate/ct_rate_vlm_phase2_fully_trained.pt \
  --filename ct_rate_vlm_phase2_fully_trained.pt
```

## Deploy The Space

The `Deploy Hugging Face Space` workflow syncs `lumora-web/backend/lumora` to the Space whenever files in that folder change on `main`. It can also be run manually from GitHub Actions.

## Model Upload Workflow

The `Upload Hugging Face Models` workflow can upload checkpoints from GitHub Actions if the checkpoint files are present in the checked-out repository, for example via Git LFS or an artifact-producing training workflow.

For normal local training, use `scripts/upload_hf_model.py` instead of committing `.pt` files.
