"""Layer-S control for W3 (C-PLAN R1 construction, audited): all-Z x 68 375 shots + the 128 even-parity X/Y patterns on 8
positions, extended periodically over the 50 qubits, x 2 547 shots each. Verified with certify (every query, every batch).
Output: derisk/results/HB1/W3_static_design.json"""
import itertools, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from shadowopt.workloads import w3_local_fidelities
from shadowopt.planner import setting_counts
from shadowopt.certify import BatchCSR, certify
from shadowopt.cost import hw1_fitted

fam = w3_local_fidelities(50)
U = fam.universe
settings = [(np.full(50, 2, np.int64), 68375)]
for bits in itertools.product((0, 1), repeat=8):
    if sum(bits) % 2:  # even number of Y (code 1) among 8 positions
        continue
    pat = np.array([1 if x else 0 for x in bits], np.int64)  # 0 = X, 1 = Y
    settings.append((np.array([pat[q % 8] for q in range(50)], np.int64), 2547))
m = setting_counts(U, settings)
worst = 0.0
for b in fam.batches:
    csr = BatchCSR(b)
    hw, ok = certify(csr, m[csr.terms])
    worst = max(worst, float(np.max(hw / csr.qeps)))
    assert np.all(ok)
cost = hw1_fitted()
U_s = cost.target([k for _, k in settings])
out = {"settings": len(settings), "shots": int(sum(k for _, k in settings)), "U_s": U_s, "worst_hw_over_eps": worst}
print(out)
json.dump(out, open(os.path.join(ROOT, "derisk", "results", "HB1", "W3_static_design.json"), "w"), indent=1)
