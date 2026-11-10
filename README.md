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
| `scripts/paper_figs.py`, `paper_figs/` | regenerates every data figure of the paper (Figs. 1, 3-5, 7-12) from the result files into `paper/figs/`; `paper_figs/illustrations.tex` holds the LaTeX of the illustrations (Fig. 2, Fig. 6, Table I); `figure_numbers.json` lists the numbers behind each figure |
| `qpu/` | hardware usage records of the cost-model jobs and their fit (`HW1_fit.json`, `ledger.csv`, `raw/HW1/`) |

## Environment

Python 3.11 with the package versions in `requirements.txt` (`pip install -r requirements.txt`). The paper's timings
ran on one workstation with an Intel Core i9-13900K (8 performance and 16 efficiency cores), 64 GB of memory and
Windows 11; the QPU-usage results are deterministic computations and do not depend on the machine. Timing scripts pin threads explicitly (`NUMBA_NUM_THREADS`,
`OMP_NUM_THREADS`) and record the thread count in every result. They run only while a lock file
(`$SHADOWOPT_LOCK_FILE`, default `CPU_LOCK.txt`) has a first line starting with `HOLDER: shadowopt`; this guard keeps
timings on a quiet machine.

## Third-party code

Three reference implementations are used as published and are not redistributed here. Clone them into `third_party/`:

```
git clone https://github.com/hsinyuan-huang/predicting-quantum-properties third_party/hkp
git -C third_party/hkp checkout dce179128f7fd91a8f9128f29be5492e8a1461bb
git clone https://gitlab.com/GreschAI/shadowgrouping third_party/shadowgrouping
git -C third_party/shadowgrouping checkout ac472cd648711ffe6ffdc3f78fd7d5169eb92cf2
git clone https://github.com/AndrewJena/VQE_measurement_optimization third_party/aequo
git -C third_party/aequo checkout 2aed9e15626f5f3988d45861b1746599f754a8d0
```

## Reproducing the paper from the frozen result files

| Paper item | How to reproduce |
|---|---|
| Figs. 1, 3, 4, 5, 7, 8, 9, 10, 11, 12 | `python scripts/paper_figs.py` (functions `fig_motivation`, `fig_select`, `fig_decisions`, `fig_costmodel`, `fig_speedup`, `fig_pathmodel`, `fig_scaling`, `fig_attribution`, `fig_perbatch`, `fig_validity`); the numbers behind each figure are written to `paper/figs/figure_numbers.json` |
| Fig. 2, Fig. 6, Table I | LaTeX in `paper_figs/illustrations.tex` (illustrations, not data) |
| Table II (batch times) | `derisk/results/D1/results.jsonl`, records `batch\|<state>\|1000000\|<path>\|T16` (median of 5) and `paired\|...` (least favourable readings); `python scripts/d1_analyze.py` recomputes `D1_summary.json` |
| Table III (space, build) | `derisk/results/D1/results.jsonl`, records with `kind` = `build` at N = 1e6 (fields `packed_bytes`, `bitmap_bytes`, `view_bytes`, `bitmap_build_s`, `view_select_s`, `view_build_s`) |
| Table IV (QPU usage) | DEV: `python scripts/hb1_harness.py --a15 --summary` (results in `derisk/results/HB1*/`); TEST: `python scripts/hb1_test_partial.py` (writes `results/HB1_TEST/partial_summary.json`, column N0 = 0) |
| Checking timings (Sec. VIII-F) | `derisk/results/HB3/results.json` |

## Re-running the experiments

All commands run from the repository root.

```
# development stores (N = 1e6 uniformly random snapshots per DEV state)
python scripts/gen_stores.py --threads 16

# answering timings (Table II, Figs. 7-9), path calibration (Fig. 8) and checking timings (Sec. VIII-F); quiet machine
python scripts/d1_timing.py
python scripts/d1_analyze.py
python scripts/pathopt_calib.py
python scripts/hb3_timing.py

# validity: decision rules (Fig. 4) and the replicate campaign (Fig. 12)
python scripts/d2_coverage.py a,b,c,d,e,f,f_alt1,f_alt2
python scripts/hc_campaign.py LiH 1000
python scripts/hc_campaign.py TFIM20 1000
python scripts/hc_campaign.py GHZ50 1000
python scripts/hc_campaign.py LiH 1000 --summary      # likewise for TFIM20 and GHZ50

# QPU usage on DEV (Table IV, Figs. 10-11): baseline workflows and controls first, then the engine runs, then the summary
python scripts/hb1_harness.py
python scripts/hb1_harness.py --a15
python scripts/hb1_harness.py --a15 --summary

# QPU usage on TEST, for example one instance with all methods
python scripts/hb1_test.py --inst=W6:Cluster100 --N0=0 --seeds=100
python scripts/hb1_test_partial.py
```

QPU-usage results are computed with the fitted hardware cost model and simulated measurement outcomes; no QPU access
is needed. The hardware cost-model jobs are recorded in `qpu/` (plan in `qpu/HW1_estimate.md`, records in
`qpu/ledger.csv` and `qpu/raw/HW1/`, calibration snapshot in `qpu/calibrations/`, fit by `scripts/hw1_fit.py`).
Re-running them (`scripts/hw1_run.py`) needs an IBM Quantum account saved locally by `qiskit_ibm_runtime` and spends
paid QPU time.

## License

MIT (see `LICENSE`). The third-party implementations keep their own licenses.
