.PHONY: setup audit train anatomy test dev dev-api dev-web build docker lint

setup:            ## install Python + frontend dependencies
	uv sync --all-groups
	cd frontend && npm ci

audit:            ## data audit -> reports/data_audit.md
	uv run python -m ml.audit

train:            ## full validation + final model -> ml/artifacts/
	uv run python -m ml.train

anatomy:          ## BodyParts3D -> frontend/public/models/anatomy.glb
	uv run python tools/build_anatomy.py

test:
	uv run pytest -q
	cd frontend && npx vitest run

lint:
	uv run ruff check ml backend tools
	cd frontend && npx tsc -b --noEmit

dev-api:
	uv run uvicorn backend.app.main:app --port 8000 --reload

dev-web:
	cd frontend && npx vite

dev:              ## API on :8000 and web on :5173
	$(MAKE) -j2 dev-api dev-web

build:
	cd frontend && npx vite build

docker:
	docker compose up --build
