import numpy as np
import pytest
from fastapi.testclient import TestClient

from datasets import generate_dataset
from forge.mlp import MLPSpec
from forge.session import TrainingSettings, create_session
from main import app

client = TestClient(app)


def test_training_is_real_and_improves():
    X, y = generate_dataset("Gaussian", 0.1)
    s = create_session(MLPSpec(2, (8,), ("ReLU",), 2), np.asarray(X), np.asarray(y), "Gaussian", seed=0)
    start = s.history[0]
    s.train(TrainingSettings(epochs=20, learning_rate=0.05, batch_size=32))
    assert [r["epoch"] for r in s.history] == list(range(21))
    assert s.history[-1]["loss"] < start["loss"]
    assert s.history[-1]["accuracy"] >= 0.95
    assert s.checkpoints.epochs[0] == 0 and s.checkpoints.epochs[-1] == 20


def test_training_is_deterministic():
    X, y = generate_dataset("XOR", 0.1)
    runs = []
    for _ in range(2):
        s = create_session(MLPSpec(2, (6,), ("Tanh",), 2), np.asarray(X), np.asarray(y), "XOR", seed=1)
        s.train(TrainingSettings(epochs=3))
        runs.append(s.history[-1]["loss"])
    assert runs[0] == runs[1]


@pytest.fixture
def sid():
    r = client.post("/api/forge/sessions", json={"neurons": [4, 3], "activations": ["ReLU", "Sigmoid"]})
    assert r.status_code == 200
    body = r.json()
    assert [layer["size"] for layer in body["structure"]["layers"]] == [2, 4, 3, 2]
    assert body["epoch"] == 0 and len(body["history"]) == 1
    return body["session_id"]


def test_full_flow(sid):
    r = client.post(f"/api/forge/sessions/{sid}/train", json={"epochs": 2})
    assert r.status_code == 200 and r.json()["summary"]["epoch"] == 2
    assert client.post(f"/api/forge/sessions/{sid}/graph", json={}).status_code == 200
    for ref in ({"kind": "neuron", "layer": 1, "index": 0}, {"kind": "layer", "layer": 2},
                {"kind": "connection", "layer": 3, "source": 2, "target": 1}):
        r = client.post(f"/api/forge/sessions/{sid}/inspect", json={"ref": ref})
        assert r.status_code == 200 and r.json()["kind"] == ref["kind"]
    r = client.post(f"/api/forge/sessions/{sid}/compare", json={
        "interventions": [{"type": "ablate_neuron", "layer": 1, "index": 0}]})
    assert r.status_code == 200 and r.json()["provenance"]["interventions_applied"] == 1


def test_errors(sid):
    assert client.post("/api/forge/sessions/nope/graph", json={}).status_code == 404
    assert client.post(f"/api/forge/sessions/{sid}/inspect",
                       json={"ref": {"kind": "neuron", "layer": 1, "index": 50}}).status_code == 422
    assert client.post(f"/api/forge/sessions/{sid}/compare", json={
        "interventions": [{"type": "ablate_neuron", "layer": 3, "index": 0}]}).status_code == 422
    assert client.post(f"/api/forge/sessions/{sid}/train", json={"epochs": 0}).status_code == 422
    assert client.post("/api/forge/sessions", json={"neurons": [500], "activations": ["ReLU"]}).status_code == 422
    assert client.post("/api/forge/sessions", json={"neurons": [4], "activations": ["Swish"]}).status_code == 422


def test_custom_dataset_session():
    X = [[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
    r = client.post("/api/forge/sessions", json={
        "neurons": [3], "activations": ["Tanh"], "custom_dataset": {"X": X, "y": [0, 1, 0, 1]}})
    assert r.status_code == 200
    body = r.json()
    assert body["structure"]["input_dim"] == 3 and body["dataset_name"] == "Custom CSV"
    n = client.post(f"/api/forge/sessions/{body['session_id']}/inspect",
                    json={"ref": {"kind": "neuron", "layer": 1, "index": 0}}).json()
    assert n["response_map"] is None  # only for 2-D inputs
    bad = client.post("/api/forge/sessions", json={"custom_dataset": {"X": X, "y": [0, 2, 0, 1]}})
    assert bad.status_code == 422


def test_capabilities_and_legacy_endpoints_still_work():
    assert client.get("/api/forge/capabilities").json()["supported_model_types"] == ["ANN"]
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.post("/api/build-network", json={}).status_code == 200
