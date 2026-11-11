# Proposed second IQM Garnet job

Prepared on 12 November 2026 for the remaining qBraid credits. Nothing here has been run on hardware: every number below is a simulation.

## The job

`job.qasm`: plain QAOA with one layer on the same instance as the June job (8 candidate sites, 2 sensors, seed 42, synthetic terrain), with the two faults of the June circuit fixed.

- The cost now includes the pairwise terms, so two sites that see the same ground are not counted twice. For two sensors this two-qubit form of the coverage objective is exact.
- The penalty weight is 1.5 × the largest single-site reward, where June's was 5 × the whole grid's priority, and the cost is rescaled. No gate angle exceeds one full turn (June's reached 344 turns).

One layer where June had two: half the gates, so less noise.

1,000 shots. At the price observed in June (30 credits per job plus 0.145 per shot) that is about 175 credits.

## Why this circuit and not XY-QAOA

`prepare.py` builds three candidates, routes each onto Garnet's 20-qubit layout and simulates them with IQM's noise model (`IQMFakeGarnet`).

| Circuit | Two-qubit gates on device | P(2 sensors), noiseless | P(2 sensors), noise model | Mean ratio, noise model |
|---|---|---|---|---|
| June job (for reference) | 163 | 0.169 | 0.153 | 0.095 |
| A: plain QAOA, fixed cost, 1 layer | 83 | 0.516 | 0.356 | 0.240 |
| B: XY-QAOA from a Dicke state, 1 layer | 194 | 1.000 | 0.310 | 0.218 |
| C: Dicke state alone, no QAOA layer | 124 | 1.000 | 0.368 | 0.243 |

For scale: uniformly random bitstrings give P(2 sensors) = 0.109 and mean ratio 0.073; a random two-sensor placement gives mean ratio 0.666.

XY-QAOA's selling point is that every shot has exactly two sensors. On this device the circuit needs 194 two-qubit gates, and the noise model predicts only 31% of shots would still have two sensors, fewer than the much shorter circuit A. Preparing the Dicke state alone already costs 124 two-qubit gates. So B would mostly measure noise; A is the circuit most likely to return a readable signal.

## How far to trust the predictions

The noise model was checked against the real June job and does not reproduce it well: the total variation distance between the measured and modelled distributions is 0.34, where 2,000-shot sampling alone would give 0.13. The real device read 1 less often than the model (0.40 against 0.45 per qubit), consistent with more relaxation than the model assumes. Treat the "noise model" column as optimistic; the ranking of the candidates rests mainly on their gate counts.

## What the job can and cannot show

- If P(2 sensors) comes back clearly above June's 0.225 and the mean ratio above June's 0.145, the corrected cost function and angles improved the hardware output. With 1,000 shots the standard error on P(2 sensors) is about 0.015.
- If it comes back near 0.109, noise dominated even at this depth.
- It cannot show quantum advantage. Even noiseless, this circuit's expected shot (0.348) is far below a random two-sensor placement (0.666) and the classical solvers (1.000).

## Files

- `prepare.py`: builds and simulates the candidates; writes `candidates.json`. Needs `pip install "iqm-client[qiskit]" qiskit-aer`.
- `make_job.py`: writes `job.qasm` from candidate A's angles.
- `job.qasm`: the circuit to submit.
- `submit.py`: submits through the qBraid SDK after showing status and cost; asks for confirmation; saves the job ID and the result.
- `score.py`: scores a result file per shot with no repair, beside the June job.

## Running it

```bash
pip install qbraid
export QBRAID_API_KEY=...
python hardware_runs/proposed_garnet_run/submit.py
python hardware_runs/proposed_garnet_run/score.py hardware_runs/proposed_garnet_run/result.json
```

Afterwards, also download the result JSON and program from the qBraid job page, as was done for June, and move everything into a dated folder (`hardware_runs/iqm_garnet_<date>/`).
