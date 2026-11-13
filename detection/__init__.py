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
from .objective import (  # noqa: E402,F401
    DEFAULT_BASELINE_MIN,
    score_detection,
    site_detection_times,
)
from .solvers import DetectionPlacement, exact_detection, greedy_detection  # noqa: E402,F401
