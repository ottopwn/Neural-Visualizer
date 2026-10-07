import os
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from forge.mlp import MLPSpec  # noqa: E402
from forge.session import create_session  # noqa: E402
from datasets import generate_dataset  # noqa: E402


@pytest.fixture
def spec():
    return MLPSpec(input_dim=2, hidden=(6, 5), activations=("Tanh", "ReLU"), n_classes=2)


@pytest.fixture
def session(spec):
    X, y = generate_dataset("Circle", 0.1)
    return create_session(spec, np.asarray(X), np.asarray(y), "Circle", seed=3)
