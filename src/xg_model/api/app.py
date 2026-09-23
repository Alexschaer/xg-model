"""HTTP API that returns the expected goals value of a shot situation."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from xg_model.api.schemas import ShotRequest, XgResponse
from xg_model.modelling.artifact import XgModel
from xg_model.serving.predictor import expected_goals

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="xG model",
    description="Expected goals for a football shot, from a model built from scratch.",
)


@lru_cache
def get_model() -> XgModel:
    """Load the trained model once and reuse it for every request."""
    return XgModel.load()


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report that the service is running."""
    return {"status": "ok"}


@app.post("/api/xg")
def predict(
    shot: ShotRequest, model: Annotated[XgModel, Depends(get_model)]
) -> XgResponse:
    """Return the expected goals value of a shot situation."""
    return XgResponse(xg=expected_goals(model, shot.to_situation()))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
