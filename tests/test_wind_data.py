"""Checks on the committed fire-season wind extract (data/wind_fire_season.npz). No network."""

import datetime
from pathlib import Path

import numpy as np
import pytest

NPZ = Path(__file__).resolve().parent.parent / "data" / "wind_fire_season.npz"
YEARS = range(2012, 2025)


def expected_hours():
    total = 0
    for y in YEARS:
        start = datetime.date(y, 2, 1)
        end = datetime.date(y, 4, 30)
        total += ((end - start).days + 1) * 24
    return total


@pytest.fixture(scope="module")
def wind():
    with np.load(NPZ) as z:
        return {k: z[k] for k in z.files}


def test_arrays_present_and_same_length(wind):
    assert set(wind) >= {"time_utc", "speed_ms", "direction_deg", "lat", "lon"}
    assert len(wind["time_utc"]) == len(wind["speed_ms"]) == len(wind["direction_deg"])


def test_length_near_expected_hour_count(wind):
    n = len(wind["time_utc"])
    exp = expected_hours()
    assert abs(n - exp) <= 0.01 * exp, (n, exp)


def test_no_nan(wind):
    assert not np.isnan(wind["speed_ms"]).any()
    assert not np.isnan(wind["direction_deg"]).any()


def test_speed_range(wind):
    s = wind["speed_ms"]
    assert (s >= 0).all() and (s < 40).all()


def test_direction_range(wind):
    d = wind["direction_deg"]
    assert (d >= 0).all() and (d < 360).all()


def test_time_strictly_increasing(wind):
    assert (np.diff(wind["time_utc"]) > 0).all()


def test_timestamps_in_fire_season_months(wind):
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    months = {
        (epoch + datetime.timedelta(hours=int(h))).month for h in wind["time_utc"]
    }
    assert months <= {2, 3, 4}


def test_years_span_2012_to_2024(wind):
    epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    first = epoch + datetime.timedelta(hours=int(wind["time_utc"][0]))
    last = epoch + datetime.timedelta(hours=int(wind["time_utc"][-1]))
    assert first.year == 2012
    assert last.year == 2024
