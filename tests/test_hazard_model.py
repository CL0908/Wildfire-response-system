"""Numpy hazard predictor (hazard_model.py): reproduces the stored samples from
data/hazard_model.json, and the feature builder has the right shape and is finite.
No scikit-learn, no network."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hazard_model as hm

DATA = os.path.join(ROOT, "data", "hazard_training_500m.npz")
MODEL = os.path.join(ROOT, "data", "hazard_model.json")


@pytest.fixture(scope="module")
def arrays():
    with np.load(DATA) as z:
        return {k: z[k] for k in z.files}


def test_sklearn_not_imported():
    assert "sklearn" not in sys.modules


def test_build_features_shape_and_finite(arrays):
    X, names = hm.build_features(arrays)
    assert X.shape == (200 * 200, len(names))
    assert 15 <= len(names) <= 20 and len(set(names)) == len(names)
    assert np.isfinite(X).all()


@pytest.mark.parametrize("name", ["validated", "deployment", "static"])
def test_predict_reproduces_stored_samples(arrays, name):
    model = hm.load_model(MODEL, name)
    assert len(model["samples"]) >= 10
    p = hm.predict(arrays, model)
    assert p.shape == (200, 200)
    assert (p >= 0).all() and (p <= 1).all()
    for s in model["samples"]:
        assert abs(p[s["row"], s["col"]] - s["p"]) < 1e-6


def test_history_models_need_matching_history(arrays):
    model = hm.load_model(MODEL, "validated")
    with pytest.raises(ValueError):
        hm.predict(arrays, model, fire_history=np.zeros((5, 5)))
    # more recent fires in a cell raise its score
    base = hm.history_counts(arrays, model)
    hi = base.copy()
    hi[100, 100] += 50
    assert hm.predict(arrays, model, hi)[100, 100] > hm.predict(arrays, model, base)[100, 100]


def test_model_file_is_complete():
    with open(MODEL) as f:
        doc = json.load(f)
    for m in doc["models"].values():
        n = len(m["features"])
        assert len(m["mean"]) == len(m["scale"]) == len(m["coef"]) == n
        assert all(s > 0 for s in m["scale"])
