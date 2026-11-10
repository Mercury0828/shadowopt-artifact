# SHADOWOPT artifact

Archived at Zenodo, DOI [10.5281/zenodo.24120997](https://doi.org/10.5281/zenodo.24120997).

Code and data for the paper *SHADOWOPT: Indexed, Certified Query Processing over Append-Only Stores of Quantum
Measurements* (Jiachen Shen and Hui Zhong, submitted to IEEE ICDE 2027).

SHADOWOPT stores randomized-measurement snapshots of a quantum state in an append-only store, answers batches of Pauli
expectation queries through compatibility bitmaps, shared intersection plans and marginal views, certifies every answer
with a Hoeffding half-width computed from counts, and plans new QPU acquisitions under a cost model fitted on hardware
usage records.

## Layout

| Path | Content |
|---|---|
| `shadowopt/` | the library: store and bit packing (`store.py`, `_bits.py`), access paths (`scan.py`, `bitmaps.py`, `sharedplan.py`, `views.py`, `pathopt.py`), certification (`bounds.py`, `certify.py`), completion planning (`planner*.py`, `alloc.py`, `derand.py`, `cost.py`), engine (`engine.py`), states and workloads (`states.py`, `states_test.py`, `workloads.py`), QPU access (`qpu.py`) |
| `scripts/` | experiment drivers: answering timings (`d1_*.py`), validity campaigns (`d2_coverage.py`, `hc_campaign.py`), QPU-usage comparison (`hb1_harness.py`, `hb1_test.py`, `hb1_test_partial.py`), checking timings (`hb3_timing.py`), path cost-model calibration (`pathopt_calib.py`), hardware cost-model runs and fit (`hw1_run.py`, `hw1_fit.py`), state and store generation |
| `tests/` | unit and equivalence tests (`pytest tests`) |
| `derisk/results/` | development-split results: answering timings (`D1`), validity (`D2`, `HC`), QPU usage (`HB1`, `HB1_A15`), checking (`HB3`), path calibration (`PATHOPT`) |
| `results/HB1_TEST/` | held-out test-split QPU-usage runs, one JSON file per instance, seed, initial-store size and method |
| `stores/states/` | prepared development and test states (state vectors, MPS or stabilizer tableaux) and exact truth values used by the tests and drivers |
| `scripts/paper_figs.py`, `paper_figs/` | regenerates every figure of the paper from the result files; `figure_numbers.json` lists the numbers behind each figure |
| `qpu/` | hardware usage records of the cost-model jobs and their fit (`HW1_fit.json`, `ledger.csv`, `raw/HW1/`) |

## Environment

Python 3.11; package versions in `requirements.txt`. Timing scripts pin threads explicitly (`NUMBA_NUM_THREADS`,
`OMP_NUM_THREADS`) and record the thread count in every result. They run only while a lock file
(`$SHADOWOPT_LOCK_FILE`, default `CPU_LOCK.txt`) has a first line starting with `HOLDER: shadowopt`; this guard keeps
timings on a quiet machine.

## Third-party code

Three reference implementations are used as published and are not redistributed here. Clone them into `third_party/`:

```
git clone https://github.com/hsinyuan-huang/predicting-quantum-properties third_party/hkp            # commit dce179128f7fd91a8f9128f29be5492e8a1461bb
git clone https://gitlab.com/GreschAI/shadowgrouping third_party/shadowgrouping                      # commit ac472cd648711ffe6ffdc3f78fd7d5169eb92cf2
git clone https://github.com/AndrewJena/VQE_measurement_optimization third_party/aequo               # commit 2aed9e15626f5f3988d45861b1746599f754a8d0
```

## Reproducing the QPU-usage results

```
python scripts/hb1_test.py --inst=W6:Cluster100 --N0=0 --seeds=100   # one test instance, all methods
python scripts/hb1_test_partial.py                                     # per-family usage ratios
```

QPU-usage results are computed with the fitted hardware cost model and simulated measurement outcomes; no QPU access
is needed. Hardware runs (`scripts/hw1_run.py`) need an IBM Quantum account saved locally by `qiskit_ibm_runtime`.

## License

MIT (see `LICENSE`). The third-party implementations keep their own licenses.
