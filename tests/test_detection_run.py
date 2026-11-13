"""Smoke test of detection.run on the real instance (quick path, nothing written)."""
import numpy as np

from detection import run


def test_quick_run_on_real_instance():
    inst = run.load_instance()
    sees = inst["sees"]
    assert sees.shape == (16, 1600)
    w = inst["window"]
    rows, cols = np.divmod(np.flatnonzero(sees.any(axis=0)), inst["shape"][1])
    assert rows.min() >= w["row0"] and rows.max() < w["row0"] + w["n_rows"]
    assert cols.min() >= w["col0"] and cols.max() < w["col0"] + w["n_cols"]

    # Pins row/col order: each site sees its own cell (a transposed mapping would not, since
    # the window is square and would still pass the bounds check above).
    import json
    with open(run.RESULTS_JSON_IN) as f:
        sites = json.load(f)["sites"]
    for s in sites:
        own = (s["row"] + w["row0"]) * inst["shape"][1] + s["col"] + w["col0"]
        assert sees[s["id"], own], s

    out = run.run_analysis(quick=True, inst=inst, log=lambda *_: None)
    h = out["results"]["headline"]
    base_h = h["baseline_h"]
    p = h["placements"]
    assert p["satellite_only"]["saved_h"] == 0.0
    for name, s in p.items():
        assert 0.0 <= s["saved_h"] <= base_h, name
    assert len(p["current"]["sites"]) == len(p["timed_exact"]["sites"]) == inst["k"]

    exact, greedy = p["timed_exact"]["train_saved_h"], p["timed_greedy"]["train_saved_h"]
    assert exact >= greedy
    for fixed in ("current", "fire_history"):
        assert exact >= p[fixed]["train_saved_h"]
    assert h["exact_solver"]["proven_optimal"] and h["exact_solver"]["matches_brute_force"]
