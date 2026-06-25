FROM python:3.13-slim

# ffmpeg: required by torchcodec (torchaudio I/O backend)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

CMD ["bash", "-lc", "echo 'speech-id image. Use docker compose to run a service.'"]