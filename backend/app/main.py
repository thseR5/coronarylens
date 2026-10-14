"""CoronaryLens API v1 (contract: backend/app/schemas.py, docs/API.md).

Run: uv run uvicorn backend.app.main:app --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from backend.app import report
from backend.app.presets import load_presets
from backend.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CohortMapResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    Preset,
    SimilarRequest,
    SimilarResponse,
    TrainingPatient,
)
from backend.app.service import Service, Superseded, artifacts_dir
from ml import config
from ml.ingest import InputError
from ml.leakage import LeakageError

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = Service(artifacts_dir())
    STATE["svc"] = svc
    STATE["presets"] = load_presets(artifacts_dir())
    STATE["config"] = config.all_configs()
    yield
    svc.close()
    STATE.clear()


app = FastAPI(title="CoronaryLens API", version="1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Superseded)
async def _superseded(request: Request, exc: Exception):
    # A newer request from the same browser tab replaced this one; the client has already moved on.
    return JSONResponse(status_code=409, content={"detail": "superseded by a newer request"})


@app.exception_handler(InputError)
@app.exception_handler(LeakageError)
async def _input_error(request: Request, exc: Exception):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def svc() -> Service:
    return STATE["svc"]


def _guard(fn, *args):
    try:
        return fn(*args)
    except (InputError, LeakageError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=svc().model.model_version)


@app.get("/api/config")
def get_config():
    return STATE["config"]


@app.get("/api/presets", response_model=list[Preset])
def presets():
    return STATE["presets"]


@app.get("/api/metrics")
def metrics():
    out = json.loads((artifacts_dir() / "metrics.json").read_text(encoding="utf-8"))
    ablation = artifacts_dir() / "ablation.json"
    if ablation.exists():  # analysis-only leakage ablation, shown in the Model performance tab
        out["ablation"] = json.loads(ablation.read_text(encoding="utf-8"))
    whatif = config.REPORTS / "whatif" / "whatif.json"  # what-if plausibility check of the served model
    if whatif.exists():
        out["whatif"] = json.loads(whatif.read_text(encoding="utf-8"))
    bench = config.REPORTS / "benchmarks"  # extra benchmarks scored on the same outer folds (ml/benchmark_extra.py)
    if bench.is_dir():
        out["benchmarks"] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(bench.glob("*.json"))}
    return out


@app.get("/api/global-explanations")
def global_explanations():
    return json.loads((artifacts_dir() / "global_explanations.json").read_text(encoding="utf-8"))


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, x_client_id: str | None = Header(default=None)):
    return _guard(svc().analyze, req.features, req.mode, x_client_id)


@app.post("/api/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, x_client_id: str | None = Header(default=None)):
    return _guard(svc().explain, req.features, x_client_id)


@app.post("/api/counterfactual", response_model=CounterfactualResponse)
def counterfactual(req: CounterfactualRequest, x_client_id: str | None = Header(default=None)):
    return _guard(svc().counterfactual, req.features, req.target, req.max_changes, x_client_id)


@app.post("/api/similar", response_model=SimilarResponse)
def similar(req: SimilarRequest):
    return _guard(svc().similar, req.features, req.k)


@app.post("/api/cohort-map", response_model=CohortMapResponse)
def cohort_map(req: ExplainRequest):
    return _guard(svc().cohort_map, req.features)


@app.get("/api/training-patient/{index}", response_model=TrainingPatient)
def training_patient(index: int):
    try:
        return svc().model.training_patient(index)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"row {index} is not in the training split") from e


@app.post("/api/report", response_class=HTMLResponse)
def make_report(req: ExplainRequest):
    s = svc()
    a = _guard(s.analyze, req.features, "f1")
    e = _guard(s.explain, req.features)
    return HTMLResponse(report.render(s.model.raw_values(req.features), a, e, s.model.fmap))


@app.get("/api/report", response_class=HTMLResponse)
def make_report_get(features: str = "{}", print_dialog: bool = Query(False, alias="print")):
    """Same report as POST, addressable by URL (features = JSON object), so the browser can open it in a new tab.
    Needed when the app runs inside a sandboxed frame (e.g. a Hugging Face Space), where writing into a blank popup
    is not possible."""
    try:
        req = ExplainRequest(features=json.loads(features))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid features: {exc}") from exc
    s = svc()
    a = _guard(s.analyze, req.features, "f1")
    e = _guard(s.explain, req.features)
    return HTMLResponse(report.render(s.model.raw_values(req.features), a, e, s.model.fmap, autoprint=print_dialog))


__all__ = ["RequestValidationError", "app"]


# Production / Docker: serve the built frontend from the same origin (registered last so /api/* wins).
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_DIST, html=True), name="web")
