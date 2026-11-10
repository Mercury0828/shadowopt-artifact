"""Generate the D1 DEV stores at N = 1e6 (N = 1e5 is the prefix), seeds per derisk/PREREG.md.

Light job: numba threads capped (default 4). Records generation throughput (D0 vi) in each manifest.
"""
import argparse
import os
import sys
import time

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import Store, packed_bytes, bitmap_bytes
from shadowopt.sampling import uniform_bases, sample_statevector, MPSSampler, sample_ghz

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATE_ID = {"LiH": 1, "TFIM20": 2, "GHZ50": 3,  # DEV
            "BeH2": 11, "H2O": 12, "N2": 13, "Heis50": 14, "Heis100": 15, "Cluster100": 16, "Clifford50": 17}  # TEST (M1-A3)
SPLIT = {k: ("DEV" if v < 10 else "TEST") for k, v in STATE_ID.items()}


def load_state(name):
    d = np.load(os.path.join(ROOT, "stores", "states", name, "state.npz")) if name != "GHZ50" else None
    if name == "LiH":
        psi = d["psi"]
        n = int(np.log2(psi.shape[0]))
        return n, (lambda B, rng: sample_statevector(psi, n, B, rng))
    if name == "TFIM20":
        tens = [d[f"B{q}"] for q in range(20)]
        s = MPSSampler(tens)
        return 20, s.sample
    if name == "GHZ50":
        return 50, sample_ghz
    d = np.load(os.path.join(ROOT, "stores", "states", name, "state.npz"))
    if name in ("BeH2", "H2O", "N2"):
        psi = d["psi"]
        n = int(np.log2(psi.shape[0]))
        return n, (lambda B, rng: sample_statevector(psi, n, B, rng))
    if name in ("Heis50", "Heis100"):
        n = int(name[4:])
        s = MPSSampler([d[f"B{q}"] for q in range(n)])
        return n, s.sample
    if name in ("Cluster100", "Clifford50"):
        import stim
        from shadowopt.sampling import StabilizerSampler
        stabs = [stim.PauliString(str(x)) for x in d["stabilizers"]]
        return len(stabs), StabilizerSampler(stabs).sample
    raise KeyError(name)


def generate(name, N, seed, chunk=100_000):
    n, sampler = load_state(name)
    rng_b = np.random.default_rng(seed)
    rng_o = np.random.default_rng(seed + 500)
    st = Store(n)
    t0 = time.perf_counter()
    for c0 in range(0, N, chunk):
        k = min(chunk, N - c0)
        B = uniform_bases(rng_b, k, n)
        O = sampler(B, rng_o)
        st.append(B, O, epoch=0, src="rand")
    dt = time.perf_counter() - t0
    return st, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--states", default="LiH,TFIM20,GHZ50")
    ap.add_argument("--N", type=int, default=1_000_000)
    a = ap.parse_args()
    numba.set_num_threads(a.threads)
    try:  # light job etiquette (shared machine): BelowNormal, optional pinning
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if os.environ.get("SHADOWOPT_CPU"):
            psutil.Process().cpu_affinity([int(os.environ["SHADOWOPT_CPU"])])
    except Exception as e:
        print("priority not lowered:", e)
    exp = int(round(np.log10(a.N)))
    for name in a.states.split(","):
        seed = 1000 * STATE_ID[name] + exp
        if os.path.exists(os.path.join(ROOT, "stores", name, f"1e{exp}", "manifest.json")):  # resumable per state
            print(name, "store exists, skipped", flush=True)
            continue
        # warm-up compile on a tiny store (not timed)
        generate(name, 64, 0, chunk=64)
        st, dt = generate(name, a.N, seed)
        man = st.save(os.path.join(ROOT, "stores", name, f"1e{exp}"), {
            "state": name, "split": SPLIT[name], "generator": "scripts/gen_stores.py", "seed_bases": seed,
            "seed_outcomes": seed + 500, "basis_distribution": "uniform", "epoch": 0, "src": "rand",
            "gen_seconds": dt, "gen_threads": a.threads, "gen_snapshots_per_s": a.N / dt,
            "packed_bytes": packed_bytes(st.n, st.N), "bitmap_bytes": bitmap_bytes(st.n, st.N),
            "prefix_rule": "the N=1e5 store is the first 1e5 rows of this store"})
        print(name, st.n, st.N, f"{dt:.1f}s", f"{a.N / dt:.0f}/s", man["sha256"][:12], flush=True)


if __name__ == "__main__":
    main()
