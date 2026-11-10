"""QPU usage and latency model (guide §P4.5, G14). Planning constants until HW1 fits them.

U(job) = n_sub * t_sub + n_settings * t_set + n_exec * t_shot,  n_exec = sum over settings of shots,
n_sub = ceil(n_exec / exec_per_subjob) (planning: sub-job split rule undocumented; 1e5 executions per sub-job).
Queue time is never part of U (N5); latency Lambda = queue + U + classical time (scenario only in Phase 0).
"""
import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CostModel:
    t_sub: float = 2.0  # s per sub-job (IBM docs: "approximately 2s per sub-job")
    t_set: float = 0.0  # s per distinct setting (parameterized basis rotations; planning)
    t_shot: float = 350e-6  # s per execution = rep_delay 250 us + circuit 100 us (IBM quick formula 2 + 0.00035 n)
    exec_per_subjob: int = 100_000  # planning value (undocumented)
    label: str = "planning (documented t_sub, quick-formula t_shot)"

    def n_sub(self, n_exec):
        return 0 if n_exec <= 0 else math.ceil(n_exec / self.exec_per_subjob)

    def job(self, n_settings, n_exec):
        """Usage of one job; zero if nothing is executed."""
        if n_exec <= 0:
            return 0.0
        return self.n_sub(n_exec) * self.t_sub + n_settings * self.t_set + n_exec * self.t_shot

    def append(self, n_add):
        """APPEND of n_add random snapshots: n_add distinct settings (bindings), one shot each."""
        return self.job(n_add, n_add)

    def target(self, shots_per_setting):
        shots = [int(s) for s in shots_per_setting if s > 0]
        return self.job(len(shots), sum(shots))

    def as_dict(self):
        return asdict(self)


def hw1_fitted(t_sub=None, t_set=None, t_shot=None):
    """Cost model with the HW1 fit (qpu/HW1_fit.json, model M2, parameterized-PUB bindings; 2026-11-07, ibm_cleveland,
    26 jobs, H-d held-out median |rel err| 1.9 %). Sub-job split rule: planning value (undocumented; HW1 jobs <= 3e4
    executions). Overrides allow the confidence-interval sensitivity runs."""
    import json, os
    f = json.load(open(os.path.join(os.path.dirname(__file__), "..", "qpu", "HW1_fit.json")))
    c = f["models"]["M2"]["constants"]
    return CostModel(t_sub=c["t_sub"] if t_sub is None else t_sub, t_set=c["t_bind"] if t_set is None else t_set,
                     t_shot=c["t_shot"] if t_shot is None else t_shot,
                     label="HW1 fit M2 (t_bind per binding; sub-job split planning)")
