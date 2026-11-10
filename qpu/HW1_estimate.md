# HW1 — cost-model fit on `ibm_cleveland` (ESTIMATE ONLY — NOTHING SUBMITTED)

Status (historical, before the run): Phase 0 spent 0 QPU seconds. Approval = one message quoting this
file's sha256 and a QPU-second cap; the submission script defaults to dry-run and refuses to submit without
`--approved HW1 --cap <seconds>` (guide §0e; `shadowopt/qpu.py` is written only after approval).

## What it decides
Fits the usage model `U(job) = n_sub · t_sub + n_settings · t_set + n_exec · t_shot` (G14 form) from `job.usage()`:
(a) is per-setting cost `t_set` zero for parameterized basis rotations (one PUB, many parameter bindings) vs. separate
circuits? (b) the per-sub-job overhead `t_sub` and when jobs split into sub-jobs; (c) `t_shot`. Answers K6 (does the
plan choice collapse to shot counting?) and gives H-d's held-out check (median absolute relative error ≤ 15 %).

## Design (12 physical qubits, one fixed 12-qubit hardware-efficient state, one calibration epoch, one session day)
Documented constants used for the prediction (labelled): `t_sub = 2 s` (IBM docs, "approximately 2s per sub-job";
documented), `t_shot = 350 µs` (IBM quick formula `2 + 0.00035·<num executions>`; documented), `t_set = 0` (planning),
one sub-job per job for ≤ 10⁵ executions (planning; IBM's job-limits page gives only "At most, 10 million executions are
allowed for Sampler jobs" and that `max_shots` / `max_experiments` "are no longer relevant or enforced").
Executions = circuits after broadcasting × shots (IBM job-limits page).

| job | settings S | shots s | mode | n_exec | predicted U (s) | role |
|---|---|---|---|---|---|---|
| A | 1 | 1 000 | single circuit | 1 000 | 2.35 | train (intercept) |
| B | 1 | 10 000 | single circuit | 10 000 | 5.50 | train (slope) |
| C | 10 | 1 000 | parameterized PUB | 10 000 | 5.50 | train (t_set) |
| D1 / D2 | 100 | 100 | parameterized / separate circuits | 10 000 each | 5.50 + 5.50 | train (t_set, PUB vs circuits) |
| E1 / E2 | 1 000 | 10 | parameterized / separate circuits | 10 000 each | 5.50 + 5.50 | train (t_set) |
| F1 / F2 | 1 000 | 1 | parameterized / separate (random bases = APPEND) | 1 000 each | 2.35 + 2.35 | train (APPEND form) |
| G | 5 000 | 1 | parameterized (random bases) | 5 000 | 3.75 | train |
| H | 300 | 30 | parameterized | 9 000 | 5.15 | **held out** |
| I | 3 000 | 1 | parameterized (random bases) | 3 000 | 3.05 | **held out** |
| J | 10 | 3 000 | parameterized | 30 000 | 12.50 | **held out** |

One pass = 64.5 s predicted; **two repetitions** (job-to-job variance) = **129 s**; with the 30 % margin = **168 s**.
**Proposed cap: 200 QPU-seconds** (planning ceiling in the guide: 300 s).

## Budget context
`service.usage()` queried 2026-11-07 08:45 UTC (no job): limit **10 800 s per period** (period 2026-10-10 → 2026-11-07
08:45 UTC, just rolled over), consumed 1 295 s, remaining 9 505 s at that instant (qpu/service_usage_20261107T084519Z.json).
The budget is shared with other hardware projects; `service.usage()` is re-queried immediately before and
after the run and logged in `qpu/ledger.csv` with every job's `job.usage()`, queue time and calibration snapshot path.

## Fit and check
Least squares on the training jobs (non-negative constants), leave-one-job-out cross-validation; H-d = median |relative
error| on jobs H, I, J ≤ 15 %. Queue time is recorded as latency only, never as usage (N5).
