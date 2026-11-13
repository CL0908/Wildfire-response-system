"""Fire-spread scenario simulation for sensor placement (simplified, uncalibrated)."""
from .fire_sim import (  # noqa: F401
    ScenarioSet,
    WindRecord,
    daytime_wind_pool,
    ignition_weights_from_counts,
    load_real_window,
    load_wind,
    simulate_scenarios,
)
