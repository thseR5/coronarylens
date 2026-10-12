.PHONY: setup audit train anatomy test test-e2e dev dev-api dev-web build serve docker lint

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

test-e2e:         ## Playwright: full stack + 3D scene (needs `npx playwright install chromium` once)
	cd frontend && npx playwright test -c e2e/app.config.ts && npx playwright test -c e2e/playwright.config.ts --project gpu

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

serve:            ## production-style: build web app, serve web + API on :8000
	cd frontend && npx vite build
	uv run uvicorn backend.app.main:app --port 8000

docker:
	docker compose up --build
