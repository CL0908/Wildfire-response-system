"""Tests for the click-to-respond precompute (ranger_response.response_explorer) and the committed
dashboards/public/response.json + response_arrival.bin."""
import json
import os

import numpy as np
import pytest

from ranger_response import experiment as ex
from ranger_response import response_explorer as re_
from ranger_response.model import Plan
from ranger_response.model2 import simulate2

PUBLIC = re_.PUBLIC


@pytest.fixture(scope="module")
def data():
    with open(re_.OUT_JSON) as f:
        meta = json.load(f)
    with open(re_.OUT_BIN, "rb") as f:
        buf = f.read()
    return meta, buf


@pytest.fixture(scope="module")
def setup():
    wind = re_.representative_wind()
    return wind, re_.base_instance(wind), re_.load_site()


def test_start_h_zero_reproduces_existing_v2_results():
    with open(ex.RESULTS_PATH) as f:
        exp = json.load(f)
    for rec in [r for r in exp["results"] if r["scenario"] in (0, 7, 15)]:
        scen = exp["scenarios"][rec["scenario"]]
        inst = ex.build_instance(scen, rec["crews"])
        assert inst.start_h == 0.0
        for m, mr in rec["methods"].items():
            plan = Plan.of([[tuple(c) for c in route] for route in mr["plan"]])
            res = simulate2(inst, plan)
            assert res.value_saved == pytest.approx(mr["value_saved"], abs=1e-9)
            assert res.cells_burned == mr["cells_burned"]
            res0 = simulate2(inst.with_params(start_h=0.0), plan)
            assert res0.burn_flat == res.burn_flat and [d.finish for d in res0.defences] == [d.finish for d in res.defences]


def test_start_h_shifts_crews_only():
    scen = dict(id=0, ignition=[10, 10], wind_speed_ms=2.0, wind_from_deg=200.0)
    inst = ex.build_instance(scen, 1)
    plan = Plan.of([[(10, 12), (10, 13)]])
    a, b = simulate2(inst, plan), simulate2(inst.with_params(start_h=3.0), plan)
    for da, db in zip(a.defences, b.defences):
        assert db.arrive == pytest.approx(da.arrive + 3.0) and db.finish == pytest.approx(da.finish + 3.0)
    base = simulate2(inst, Plan.empty(1))
    assert simulate2(inst.with_params(start_h=3.0), Plan.empty(1)).burn_flat == base.burn_flat   # fire unchanged
    assert inst.with_params(start_h=2.5).delay_h == 2.5
    with pytest.raises(ValueError):
        inst.with_params(start_h=-1.0)


def test_detection_is_min_arrival_over_seen_cells_and_ties_go_to_lowest_site(setup):
    wind, base, site = setup
    for ig in [(0, 0), (19, 19), (13, 15), (6, 12), (10, 10), (2, 5)]:
        arr = np.array(simulate2(re_.instance_for(base, ig, 1, 0.0), Plan.empty(1)).burn_flat)
        seen = np.unique(np.concatenate([site["sees"][j] for j in site["chosen"]]))
        t = float(arr[seen].min())
        det = re_.detection(base, site, ig, arr)
        assert det["detection_h"] == pytest.approx(min(t, re_.SATELLITE_H))
        assert det["sensor_h"] == (t if np.isfinite(t) else None)
        if t < re_.SATELLITE_H:
            assert det["per_site_h"][det["detecting_site"]] == t
            assert all(j >= det["detecting_site"] for j, v in det["per_site_h"].items() if v == t)
        else:
            assert det["detecting_site"] == "satellite"
    # tie: two sites with the same viewshed -> lowest id
    fake = dict(sees={7: np.array([5, 6]), 3: np.array([5, 6])})
    arr = np.full(400, np.inf)
    arr[5], arr[6] = 1.5, 2.0
    d = re_.detection(base, fake, (0, 0), arr)
    assert d["detecting_site"] == 3 and d["detection_h"] == 1.5
    arr[5] = arr[6] = 7.0      # sensors only see it after the satellite
    assert re_.detection(base, fake, (0, 0), arr)["detecting_site"] == "satellite"


def test_exported_detection_matches_recomputation(data, setup):
    meta, _ = data
    wind, base, site = setup
    assert meta["parameters"]["wind"]["value"]["speed_ms"] == wind["speed_ms"]
    assert meta["sensors"]["site_ids"] == site["chosen"]
    for i in (0, 57, 123, 211, 300, 399):
        ig = divmod(i, 20)
        det = re_.detection(base, site, ig)
        rec = meta["ignitions"][i]
        assert rec["cell"] == list(ig)
        assert rec["modes"]["sensors"]["detection_h"] == pytest.approx(det["detection_h"], abs=1e-3)
        assert rec["modes"]["sensors"]["detecting_site"] == det["detecting_site"]
        assert rec["modes"]["sensors"]["start_h"] == pytest.approx(det["detection_h"] + 0.5, abs=1e-3)


def test_satellite_plans_never_start_before_6h30(data):
    meta, _ = data
    for rec in meta["ignitions"]:
        sat = rec["modes"]["satellite"]
        assert sat["detecting_site"] == "satellite" and sat["detection_h"] == 6.0 and sat["start_h"] == 6.5
        for crews in (1, 2):
            for d in rec["plans"][f"satellite_{crews}"]["defences"]:
                assert d["arrive_h"] >= 6.5 - 1e-9
        m = rec["modes"]["sensors"]
        for crews in (1, 2):
            for d in rec["plans"][f"sensors_{crews}"]["defences"]:
                assert d["arrive_h"] >= m["start_h"] - 1e-2


def test_sensors_never_worse_than_satellite(data):
    meta, _ = data
    for rec in meta["ignitions"]:
        for crews in (1, 2):
            assert rec["plans"][f"sensors_{crews}"]["value_saved"] >= rec["plans"][f"satellite_{crews}"]["value_saved"] - 1e-3


def test_bin_round_trip_matches_simulate(data, setup):
    meta, buf = data
    wind, base, site = setup
    assert len(buf) == 400 * 5 * 400 == meta["arrival_bin"]["bytes"]
    for i in (3, 150, 222, 399):
        rec = meta["ignitions"][i]
        ig = tuple(rec["cell"])
        for key in re_.PLAN_KEYS:
            if key == "no_response":
                inst, plan = re_.instance_for(base, ig, 1, 0.0), Plan.empty(1)
            else:
                mode, crews = key.split("_")
                crews = int(crews)
                inst = re_.instance_for(base, ig, crews, rec["modes"][mode]["start_h"])
                routes = [[] for _ in range(crews)]
                for d in sorted(rec["plans"][key]["defences"], key=lambda d: (d["crew"], d["order"])):
                    routes[d["crew"]].append(tuple(d["cell"]))
                plan = Plan.of(routes)
            res = simulate2(inst, plan)
            truth = res.burn_time
            dec = re_.decode_arrival(buf, i, key)
            assert np.array_equal(np.isfinite(truth), np.isfinite(dec))
            assert np.allclose(dec[np.isfinite(dec)], truth[np.isfinite(truth)], atol=0.051)
            if key != "no_response":
                assert res.value_saved == pytest.approx(rec["plans"][key]["value_saved"], abs=2e-3)
                assert res.cells_burned == rec["plans"][key]["cells_burned"]
                assert int(np.isfinite(dec).sum()) == res.cells_burned


def test_quantise_edges():
    q = re_.quantise(np.array([0.0, 0.04, 0.06, 23.99, 24.0, np.inf]))
    assert list(q) == [0, 0, 1, 240, 240, 255]


def test_json_schema_and_size(data):
    meta, buf = data
    for k in ("model", "grid", "arrival_bin", "parameters", "sensors", "assumptions", "ignitions", "aggregate", "sanity_check", "caveats"):
        assert k in meta
    assert meta["grid"] == dict(rows=20, cols=20, cell_size_m=500, horizon_h=24.0)
    assert meta["arrival_bin"]["plans"] == list(re_.PLAN_KEYS) and meta["arrival_bin"]["layout"]
    for k in ("wind", "satellite_detection_h", "dispatch_delay_h", "detection_rule", "cut_h_per_cell", "solver"):
        assert k in meta["parameters"]
    assert meta["parameters"]["dispatch_delay_h"]["value"] == 0.5 and meta["parameters"]["satellite_detection_h"]["value"] == 6.0
    assert any("not proven optimal" in a for a in meta["assumptions"])
    ign = meta["ignitions"]
    assert [r["id"] for r in ign] == list(range(400)) and all(r["cell"] == [i // 20, i % 20] for i, r in enumerate(ign))
    sites = set(meta["sensors"]["site_ids"])
    for r in ign:
        assert set(r) == {"id", "cell", "station", "sensor_first_h", "baseline", "modes", "plans"}
        assert set(r["modes"]) == {"sensors", "satellite"} and set(r["plans"]) == {"sensors_1", "sensors_2", "satellite_1", "satellite_2"}
        assert r["modes"]["sensors"]["detecting_site"] in sites | {"satellite"}
        for p in r["plans"].values():
            assert {"value_saved", "cells_burned", "pct_saved", "defences"} <= set(p)
            assert 0 <= p["pct_saved"] <= 100 + 1e-6
            for d in p["defences"]:
                assert set(d) == {"crew", "order", "cell", "arrive_h", "finish_h", "fire_arrival_h", "held"}
    agg = meta["aggregate"]
    assert agg["n_ignitions"] == 400 and 0 < agg["detection"]["sensors_detect_share"] <= 1
    assert len(meta["sanity_check"]["rows"]) == 10
    total = os.path.getsize(re_.OUT_JSON) + os.path.getsize(re_.OUT_BIN)
    assert total < 2_000_000
