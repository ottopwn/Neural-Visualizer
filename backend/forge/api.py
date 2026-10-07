"""HTTP API for Neural Forge (mounted under ``/api/forge``)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from datasets import generate_dataset

from . import schema as S
from .checkpoints import DEFAULT_CAPACITY
from .introspect import IntrospectionError, MLPIntrospector
from .landscape import LossLandscape, loss_landscape
from .mlp import SUPPORTED_ACTIVATIONS, MLPSpec
from .session import ModelSession, SessionRegistry, TrainingSettings, create_session
from .timemachine import TimeMachine

router = APIRouter(prefix="/api/forge", tags=["forge"])
registry = SessionRegistry(max_sessions=8)

MAX_NEURONS = 128
MAX_LAYERS = 5
MAX_SAMPLES = 2000
MAX_FEATURES = 16
MAX_EPOCHS_PER_CALL = 500


class CustomDataset(BaseModel):
    X: List[List[float]]
    y: List[int]


class CreateSessionRequest(BaseModel):
    neurons: List[int] = Field(default_factory=lambda: [8, 8])
    activations: List[str] = Field(default_factory=lambda: ["ReLU", "ReLU"])
    dataset: str = "Circle"
    noise: float = 10.0
    custom_dataset: Optional[CustomDataset] = None
    seed: int = 0


class TrainRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    reg_type: str = "None"
    reg_rate: float = 0.0


class TrainResponse(BaseModel):
    summary: S.SessionSummary
    new_rows: List[S.HistoryRow]


def _summary(session: ModelSession) -> S.SessionSummary:
    return S.SessionSummary(
        session_id=session.id,
        structure=MLPIntrospector(session).structure(),
        dataset_name=session.dataset_name,
        dataset_X=session.X.tolist(),
        dataset_y=[int(v) for v in session.y.tolist()],
        epoch=session.epoch,
        history=[S.HistoryRow(**r) for r in session.history],
        checkpoints=[S.HistoryRow(**r) for r in session.checkpoints.summary()],
    )


def _get(session_id: str) -> ModelSession:
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown or expired session — rebuild the network.")
    return session


def _bad(exc: Exception) -> HTTPException:
    return HTTPException(422, str(exc))


@router.get("/capabilities")
def capabilities():
    """What the introspection layer can do, so the UI never over-promises."""
    return {
        "supported_model_types": ["ANN"],
        "activations": list(SUPPORTED_ACTIVATIONS),
        "interventions": ["ablate_neuron", "set_weight", "set_bias"],
        "pass_explorer": {"trace": True, "sgd_preview": True},
        "time_machine": {"checkpoint_capacity": DEFAULT_CAPACITY, "interventions_in_history": False},
        "limits": {"max_neurons": MAX_NEURONS, "max_layers": MAX_LAYERS},
        "unsupported_note": (
            "CNN, RNN, LSTM, GAN, Transformer and Diffuser graphs are illustrative "
            "diagrams; their node values are not computed from a model."
        ),
    }


@router.post("/sessions", response_model=S.SessionSummary)
def create(req: CreateSessionRequest):
    if not 1 <= len(req.neurons) <= MAX_LAYERS or len(req.neurons) != len(req.activations):
        raise _bad(ValueError(f"need 1..{MAX_LAYERS} hidden layers with one activation each"))
    if any(not 1 <= n <= MAX_NEURONS for n in req.neurons):
        raise _bad(ValueError(f"neurons per layer must be in 1..{MAX_NEURONS}"))

    if req.custom_dataset is not None:
        X = np.asarray(req.custom_dataset.X, dtype=np.float32)
        y = np.asarray(req.custom_dataset.y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y) or not 2 <= len(X) <= MAX_SAMPLES:
            raise _bad(ValueError(f"custom dataset must be a 2-D table with 2..{MAX_SAMPLES} rows"))
        if not 1 <= X.shape[1] <= MAX_FEATURES or not np.isfinite(X).all():
            raise _bad(ValueError(f"custom dataset needs 1..{MAX_FEATURES} finite feature columns"))
        if set(np.unique(y).tolist()) - {0, 1}:
            raise _bad(ValueError("custom dataset labels must be 0/1"))
        name = "Custom CSV"
    else:
        Xl, yl = generate_dataset(req.dataset, req.noise / 100.0)
        X, y = np.asarray(Xl, dtype=np.float32), np.asarray(yl, dtype=np.int64)
        name = req.dataset

    try:
        spec = MLPSpec(input_dim=int(X.shape[1]), hidden=tuple(req.neurons),
                       activations=tuple(req.activations), n_classes=2)
    except ValueError as exc:
        raise _bad(exc)
    session = create_session(spec, X, y, name, seed=req.seed)
    registry.add(session)
    return _summary(session)


@router.get("/sessions/{session_id}", response_model=S.SessionSummary)
def get_session(session_id: str):
    return _summary(_get(session_id))


@router.post("/sessions/{session_id}/train", response_model=TrainResponse)
def train(session_id: str, req: TrainRequest):
    session = _get(session_id)
    if not 1 <= req.epochs <= MAX_EPOCHS_PER_CALL:
        raise _bad(ValueError(f"epochs must be in 1..{MAX_EPOCHS_PER_CALL}"))
    if not 0 < req.learning_rate <= 1 or req.batch_size < 1 or req.reg_rate < 0:
        raise _bad(ValueError("invalid learning rate / batch size / regularisation rate"))
    with session.lock:
        rows = session.train(TrainingSettings(
            epochs=req.epochs, learning_rate=req.learning_rate, batch_size=req.batch_size,
            reg_type=req.reg_type, reg_rate=req.reg_rate, seed=session.seed,
        ))
    return TrainResponse(summary=_summary(session), new_rows=[S.HistoryRow(**r) for r in rows])


def _with_session(session_id: str, fn):
    session = _get(session_id)
    with session.lock:
        try:
            return fn(MLPIntrospector(session))
        except IntrospectionError as exc:
            raise _bad(exc)


@router.post("/sessions/{session_id}/graph", response_model=S.ForgeGraph)
def graph(session_id: str, req: S.ExperimentRequest):
    return _with_session(session_id, lambda i: i.graph(req))


@router.post("/sessions/{session_id}/inspect",
             response_model=S.NeuronInspection | S.LayerInspection | S.ConnectionInspection)
def inspect(session_id: str, req: S.InspectRequest):
    return _with_session(session_id, lambda i: i.inspect(req))


@router.post("/sessions/{session_id}/trace", response_model=S.ComputationTrace)
def trace(session_id: str, req: S.TraceRequest):
    """Every forward tensor and backward gradient of one probe (Pass Explorer, 3D view)."""
    return _with_session(session_id, lambda i: i.trace(req))


@router.post("/sessions/{session_id}/compare", response_model=S.Comparison)
def compare(session_id: str, req: S.ExperimentRequest):
    return _with_session(session_id, lambda i: i.compare(req))


class LandscapeRequest(BaseModel):
    checkpoint_epoch: Optional[int] = None


@router.post("/sessions/{session_id}/loss-landscape", response_model=LossLandscape)
def landscape(session_id: str, req: LandscapeRequest):
    """A real 2-D loss slice around the live weights or a stored checkpoint."""
    session = _get(session_id)
    with session.lock:
        try:
            return loss_landscape(session, req.checkpoint_epoch)
        except KeyError as exc:
            raise _bad(exc)


# ── Training Time Machine ───────────────────────────────────────────────────
# Read-only views of stored checkpoints; interventions are never applied.

def _with_time_machine(session_id: str, fn):
    session = _get(session_id)
    with session.lock:
        try:
            return fn(TimeMachine(session))
        except IntrospectionError as exc:
            raise _bad(exc)


@router.get("/sessions/{session_id}/timeline", response_model=S.Timeline)
def timeline(session_id: str):
    """Training log, per-checkpoint health statistics, runs and notable events."""
    return _with_time_machine(session_id, lambda tm: tm.timeline())


@router.post("/sessions/{session_id}/frame", response_model=S.Frame)
def frame(session_id: str, req: S.FrameRequest):
    """The network at one stored checkpoint: metrics, predictions, decision boundary."""
    return _with_time_machine(session_id, lambda tm: tm.frame(req))


@router.post("/sessions/{session_id}/component-history", response_model=S.ComponentHistory)
def component_history(session_id: str, req: S.ComponentHistoryRequest):
    """One neuron / layer / connection across every stored checkpoint."""
    return _with_time_machine(session_id, lambda tm: tm.component_history(req))


@router.post("/sessions/{session_id}/epoch-compare", response_model=S.EpochComparison)
def epoch_compare(session_id: str, req: S.EpochCompareRequest):
    """Real differences between two stored checkpoints (A -> B)."""
    return _with_time_machine(session_id, lambda tm: tm.compare(req))
