FROM python:3.12-slim

RUN pip install uv

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user pyproject.toml .
COPY --chown=user uv.lock .

RUN uv sync --frozen --no-dev

COPY --chown=user . .

ENV HF_HOME=/tmp/hf-cache
ENV TORCH_HOME=/tmp/torch-cache
ENV TRANSFORMERS_CACHE=/tmp/transformers-cache

EXPOSE 7860
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]