# IQM Garnet run, 1 June 2026

The one real-hardware run with surviving raw data. Exported from qBraid on 12 November 2026.

| | |
|---|---|
| Job ID | `aws:iqm:qpu:garnet-eecc-qjob-6a1d097d3d86f02c1354cc36` |
| Device | IQM Garnet (20 qubits), through AWS Braket and qBraid |
| Created | 1 June 2026, 04:24:32 UTC (12:24 local) |
| Duration | 54.5 s |
| Shots | 2,000 |
| Cost | 320 qBraid credits |

## Files

- `program.qasm`: the circuit that was submitted (OpenQASM 3).
- `result.json`: status, timestamps and the measurement counts for all 2,000 shots.
- `circuit_diagram.txt`: the circuit drawn by Qiskit from `program.qasm`.
- `qbraid_job_details.png`: screenshot of the job page.
- `rescore.py`: scores the counts against the instance, with no classical repair.

## What the circuit is

Plain QAOA with two layers on the smallest benchmark instance: 8 candidate sites, 2 sensors, seed 42, synthetic terrain. One qubit per candidate site.

1. `h` on every qubit: an equal superposition of all 256 placements.
2. Cost layer: `rzz` on all 28 qubit pairs with one shared angle, then `rz` on each qubit with its own angle.
3. Mixer layer: `rx` on every qubit with one shared angle.
4. Steps 2 and 3 again with the second layer's angles, then measure.

The 28 identical `rzz` gates are the penalty for placing the wrong number of sensors, `P·(number placed − 2)²`. The per-qubit `rz` angles carry the reward. There is no pairwise term between sites that see the same ground, so the circuit rewards each site for everything it sees and counts overlapping coverage twice. This is the cost function `solvers/qaoa.py` had before commit `11c32d8`.

Reconstructed from the angles (the `rzz` angles match the old code exactly):

- Penalty weight P = 679.70, which is 5 × the total priority of the grid.
- γ = 0.804 and 0.337; β = 0.623 and 0.051.
- The reward angles are 2.5 times what the old code in this repo produces for this instance. The factor is unexplained; the job was probably built from an earlier local version of the code.

Because P is so large, the cost angles are hundreds to thousands of radians (546.46 rad is 87 full turns; the largest, 2,161.53 rad, is 344). The circuit depends on those angles modulo 2π, so it is very sensitive to small changes in γ.

Depth 27 and 56 `rzz` gates agree with the "transpiled depth 27" and "|ZZ| = 28" in the pitch deck's table for this run.

## Result, scored per shot with no repair

From `python hardware_runs/iqm_garnet_2026-06-01/rescore.py`:

| Distribution | P(exactly 2 sensors) | P(optimal placement) | Mean ratio to optimum |
|---|---|---|---|
| IQM Garnet, measured | 0.225 | 0.0050 | 0.145 |
| Same circuit, noiseless simulation | 0.169 | 0.0000 | 0.104 |
| Uniform over all 256 bitstrings | 0.109 | 0.0039 | 0.073 |
| Uniform over the 28 two-sensor placements | 1.000 | 0.0357 | 0.666 |

"Mean ratio" counts a shot with the wrong number of sensors as 0.

- The optimal placement (sites 0 and 6, objective 63.998) came up in 10 of 2,000 shots. Guessing a two-sensor placement at random would hit it about 71 times in 2,000.
- The pitch reported "ratio 1.00". That was the best of 2,000 shots after classical repair; with 28 possible placements any sampler finds the optimum that way.
- The noiseless simulation of this circuit puts almost no probability on the optimum, so the circuit itself was poor before noise came into it.
- The device's output is closer to uniform noise than to the noiseless circuit (total variation distance 0.33 against 0.59). It still carries signal: the per-qubit frequencies correlate with the simulation at 0.73.
- 22.5% of shots had exactly two sensors, more than the noiseless circuit gives. Measured qubits read 1 about 40% of the time on average, which favours low counts; that is consistent with qubits relaxing towards 0, not with the circuit enforcing the constraint.

This run shows no quantum advantage, and should not be presented as recovering the optimum.

## Reading the bitstrings

The keys in `result.json` are read with qubit 0 as the rightmost character (Qiskit's order). This was inferred, not documented: per-qubit frequencies correlate with the noiseless simulation at +0.73 in this order and −0.35 in the reverse order. The share of two-sensor shots does not depend on the order; which placements they are does.

## Differences from the pitch deck

The deck reports 453 of 2,000 feasible shots for a Garnet run; these counts give 449. Another slide gives 251 of 2,000 for the same settings. The deck may describe a different job. `solvers/check.py` holds the ID of a separate job on IQM Emerald, whose results are not in this repo.
