# IQM Garnet run, 12 November 2026

The follow-up to the June job: the same instance with the cost function fixed. Prepared in `../proposed_garnet_run/`, which explains why this circuit was chosen.

| | |
|---|---|
| Job ID | `aws:iqm:qpu:garnet-eecc-qjob-6af54b2180fe3a606c20f114` |
| Device | IQM Garnet (20 qubits), through AWS Braket and qBraid |
| Created | 12 November 2026, 05:14:13 UTC (13:14 local) |
| Duration | 4.0 s |
| Shots | 1,000 |
| Cost | 175 qBraid credits |

## Files

- `program.qasm`: the circuit, a copy of `../proposed_garnet_run/job.qasm`. It was submitted with `submit.py`; the program was not exported from the qBraid job page.
- `result.json`: status, timestamps and measurement counts, downloaded from the qBraid job page.
- `submitted_job.json`: the job ID as saved by `submit.py`.

## What the circuit is

Plain QAOA with one layer on 8 candidate sites, 2 sensors, seed 42, synthetic terrain. Against June's circuit:

- The cost includes the pairwise terms, so ground seen by two sites is not counted twice.
- The penalty weight is 1.5 × the largest single-site reward and the cost is rescaled; the largest gate angle is 1.22 rad (June's reached 344 full turns).
- One layer where June had two.

## Result, scored per shot with no repair

From `python hardware_runs/proposed_garnet_run/score.py hardware_runs/iqm_garnet_2026-11-12/result.json`:

| Distribution | P(exactly 2 sensors) | P(optimal placement) | Mean ratio to optimum |
|---|---|---|---|
| This job, measured | 0.388 | 0.0280 | 0.284 |
| This job, noiseless simulation | 0.516 | 0.0202 | 0.348 |
| June job, measured | 0.225 | 0.0050 | 0.145 |
| Uniform over all 256 bitstrings | 0.109 | 0.0039 | 0.073 |
| Uniform over the 28 two-sensor placements | 1.000 | 0.0357 | 0.666 |

"Mean ratio" counts a shot with the wrong number of sensors as 0. The standard error on P(exactly 2 sensors) is 0.015.

- The device followed the circuit this time. The measured distribution is closer to the noiseless circuit than to uniform noise (total variation distance 0.32 against 0.52); in June it was the other way round (0.59 against 0.33).
- 38.8% of shots placed exactly 2 sensors, against 22.5% in June. The noise model predicted 35.6% and the noiseless circuit gives 51.6%.
- The mean ratio roughly doubled, from 0.145 to 0.284. The noise model predicted 0.240.
- The optimal placement came up in 28 of 1,000 shots. Picking a two-sensor placement at random would hit it about 36 times in 1,000.
- The average shot (0.284) is still far below a random two-sensor placement (0.666) and the classical solvers (1.000).

Fixing the cost function and shortening the circuit improved the hardware output. The run shows no quantum advantage.

## Reading the bitstrings

Qubit 0 is the rightmost character, as in June. Per-qubit frequencies correlate with the noiseless simulation at +0.78 in this order and +0.52 in the reverse order.
