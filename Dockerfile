FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app/xg-model

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/home/app/xg-model/.venv/bin:$PATH"

COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=app src ./src
COPY --chown=app models ./models
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["sh", "-c", "uvicorn xg_model.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]