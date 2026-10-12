# CoronaryLens: one container serving the API and the built web app on :8000.
# Trained artifacts and anatomy.glb are committed, so no training or mesh download happens here.

FROM node:22-slim AS web
WORKDIR /app
COPY config ./config
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci --no-audit --no-fund
COPY frontend ./frontend
RUN cd frontend && npx vite build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
# runtime groups only (sprint learners are needed to load the served model; dev/anatomy tools are not)
RUN uv sync --frozen --no-install-project --no-default-groups --group sprint
COPY config ./config
COPY ml ./ml
COPY backend ./backend
COPY data ./data
COPY --from=web /app/frontend/dist ./frontend/dist
# non-root user (required by Hugging Face Spaces; harmless elsewhere)
RUN useradd -m -u 1000 user
USER user
EXPOSE 8000
HEALTHCHECK CMD /app/.venv/bin/python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["/app/.venv/bin/uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
