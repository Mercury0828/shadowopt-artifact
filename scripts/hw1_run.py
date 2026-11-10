"""HW1 runner (guide §0e). DRY-RUN BY DEFAULT. Submits only with `--submit --approved HW1 --cap <s>` (cap <= 200, the
owner's approval of 2026-11-07, DESIGN_DECISIONS #11) and only if qpu/HW1_estimate.md still has the approved sha256.

Resumable: every finished job is a row of qpu/ledger.csv (keyed by grid name + repetition); a submitted-but-unrecorded job
id is kept in qpu/HW1_pending.json and retrieved on restart instead of resubmitting. Stops before any job whose predicted
usage (x 1.3 margin) would push measured usage past the cap. The API token is read from the saved account and never
printed or written.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import uuid

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from shadowopt.qpu import (GRID, REPETITIONS, HW1_ESTIMATE_SHA256, HW1_APPROVED_CAP_S, file_sha256, predicted_usage,
                           grid_prediction, state_circuit, binding_values, job_settings)

QPU = os.path.join(ROOT, "qpu")
LEDGER = os.path.join(QPU, "ledger.csv")
PENDING = os.path.join(QPU, "HW1_pending.json")
RAW = os.path.join(QPU, "raw", "HW1")
CAL = os.path.join(QPU, "calibrations")
FIELDS = ["date_utc", "job_id", "tier", "purpose", "role", "mode", "settings", "shots", "n_exec", "n_pubs",
          "predicted_U_s", "usage_quantum_s", "metrics_usage", "span_sum_s", "span_wall_s", "queue_s", "running_s",
          "settings_seed", "layout", "calibration_snapshot", "raw_path", "status", "notes"]
MARGIN = 1.3


def now():
    return dt.datetime.now(dt.timezone.utc)


def ledger_rows():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def measured_so_far():
    """Billed usage of every recorded HW1 job (ok or failed) + 1.3 x predicted for unresolved pending records."""
    billed = sum(float(r["usage_quantum_s"]) for r in ledger_rows()
                 if r["tier"] == "HW1" and r["usage_quantum_s"] not in ("", "None"))
    pend = json.load(open(PENDING)) if os.path.exists(PENDING) else {}
    return billed + sum(MARGIN * float(v.get("predicted", 0)) for v in pend.values())


def plan_order():
    """(name, rep) in a seeded shuffled order per repetition (spreads drift over the grid)."""
    out = []
    for rep in range(1, REPETITIONS + 1):
        idx = np.random.default_rng(1000 + rep).permutation(len(GRID))
        out += [(GRID[i], rep) for i in idx]
    return out


def build_pubs(isa, job, rep):
    name, S, s, mode, role = job
    settings, seed = job_settings(name, rep, S)
    params = list(isa.parameters)
    vals = binding_values(settings, params)
    if mode == "param":
        pubs = [(isa, vals)]
    elif mode == "single":
        pubs = [(isa.assign_parameters(dict(zip(params, vals[0]))),)]
    elif mode == "circ":
        pubs = [(isa.assign_parameters(dict(zip(params, v))),) for v in vals]
    else:
        raise ValueError(mode)
    return pubs, settings, seed


def save_raw(job_id, name, rep, settings, result, mode):
    os.makedirs(RAW, exist_ok=True)
    bits = []
    for pub_res in result:
        reg = next(iter(pub_res.data.values())) if hasattr(pub_res.data, "values") else pub_res.data.c
        arr = reg.array  # (..., shots, nbytes) uint8
        bits.append(np.asarray(arr))
    path = os.path.join(RAW, f"{name}_r{rep}_{job_id}.npz")
    np.savez_compressed(path, settings=settings, mode=mode, **{f"pub{i}": b for i, b in enumerate(bits)})
    return os.path.relpath(path, ROOT).replace("\\", "/")


def spans(result):
    try:
        sp = result.metadata["execution"]["execution_spans"]
        starts = [x.start for x in sp]
        stops = [x.stop for x in sp]
        dsum = sum((b - a).total_seconds() for a, b in zip(starts, stops))
        wall = (max(stops) - min(starts)).total_seconds() if starts else None
        return dsum, wall
    except Exception as e:  # recorded, never fatal
        return None, f"spans unavailable: {type(e).__name__}"


def metrics_times(job):
    try:
        m = job.metrics()
        ts = m.get("timestamps", {})
        f = lambda k: dt.datetime.fromisoformat(ts[k].replace("Z", "+00:00")) if ts.get(k) else None
        c, r, fi = f("created"), f("running"), f("finished")
        q = (r - c).total_seconds() if (c and r) else None
        run = (fi - r).total_seconds() if (r and fi) else None
        return json.dumps(m.get("usage", {})), q, run
    except Exception as e:
        return f"metrics unavailable: {type(e).__name__}", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="really submit (default: dry run)")
    ap.add_argument("--approved", default="")
    ap.add_argument("--cap", type=float, default=0.0)
    ap.add_argument("--new-epoch-ok", action="store_true", help="continue after a calibration update (recorded)")
    a = ap.parse_args()

    sha = file_sha256(os.path.join(QPU, "HW1_estimate.md"))
    pred = grid_prediction()
    print(f"HW1 grid: {len(GRID)} jobs x {REPETITIONS} reps; predicted {pred['all_reps_s']:.1f} s "
          f"(+30 % = {pred['with_margin_s']:.1f} s); estimate sha256 {'OK' if sha == HW1_ESTIMATE_SHA256 else 'MISMATCH'}")
    done = {(r["purpose"]) for r in ledger_rows() if r["tier"] == "HW1" and r["status"] == "ok"}
    order = [(j, rep) for j, rep in plan_order() if f"{j[0]}_r{rep}" not in done]
    print(f"remaining jobs: {len(order)}; measured HW1 usage so far: {measured_so_far():.1f} s")

    if not a.submit:
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit import transpile
        qc, _, _ = state_circuit()
        be = GenericBackendV2(num_qubits=20, basis_gates=["cz", "id", "rz", "sx", "x"], seed=1)
        isa = transpile(qc, be, optimization_level=1, seed_transpiler=11)
        for j, rep in order:
            pubs, settings, seed = build_pubs(isa, j, rep)
            print(f"  DRY {j[0]}_r{rep}: mode {j[3]}, S={j[1]}, s={j[2]}, pubs={len(pubs)}, "
                  f"pred {predicted_usage(j[1], j[2]):.2f} s, settings seed {seed}")
        print("dry run only; nothing submitted")
        return

    if a.approved != "HW1" or not (0 < a.cap <= HW1_APPROVED_CAP_S):
        sys.exit(f"refused: needs --approved HW1 --cap <= {HW1_APPROVED_CAP_S}")
    if sha != HW1_ESTIMATE_SHA256:
        sys.exit("refused: qpu/HW1_estimate.md differs from the approved version")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    service = QiskitRuntimeService()
    backend = service.backend("ibm_cleveland")
    ts = now().strftime("%Y%m%dT%H%M%SZ")
    u0 = service.usage()
    json.dump(u0, open(os.path.join(QPU, f"service_usage_{ts}_HW1_before.json"), "w"), default=str, indent=1)
    print("usage before:", {k: u0.get(k) for k in ("usage_consumed_seconds", "usage_limit_seconds", "usage_remaining_seconds")})
    pending = load_pending()

    # 1) reconcile every pending record first (no new submission before all are resolved)
    for key in list(pending):
        rec = pending[key]
        job = service.job(rec["job_id"]) if rec.get("job_id") else None
        if job is None:  # submission intent without a recorded id: search by the unique tag
            found = service.jobs(job_tags=[rec["tag"]], limit=5)
            if len(found) > 1:
                sys.exit(f"refused: {len(found)} jobs carry tag {rec['tag']}; reconcile by hand")
            if not found:
                if (now() - dt.datetime.fromisoformat(rec["intent_utc"])).total_seconds() < 600:
                    sys.exit(f"{key}: intent < 10 min old and no job found yet; retry later")
                print(f"{key}: no job carries tag {rec['tag']} after 10 min -> never submitted; intent dropped")
                pending.pop(key)
                save_pending(pending)
                continue
            job = found[0]
            rec["job_id"] = job.job_id()
            save_pending(pending)
        if not finish(job, key, rec, pending):
            sys.exit(f"{key}: job {job.job_id()} unresolved (status/billing pending); rerun later, nothing new submitted")

    # 2) calibration epoch and transpilation for new submissions
    props = backend.properties()
    epoch = str(props.last_update_date)
    first_epoch = next((r["notes"].split("epoch=")[1].split(";")[0] for r in ledger_rows()
                        if r["tier"] == "HW1" and "epoch=" in r["notes"]), None)
    if first_epoch and first_epoch != epoch and not a.new_epoch_ok:
        sys.exit(f"refused: calibration epoch changed ({first_epoch} -> {epoch}); rerun with --new-epoch-ok to continue "
                 "(the deviation is recorded)")
    os.makedirs(CAL, exist_ok=True)
    cal_path = os.path.join(CAL, f"ibm_cleveland_{ts}_HW1_properties.json")
    json.dump(props.to_dict(), open(cal_path, "w"), default=str)
    cal_rel = os.path.relpath(cal_path, ROOT).replace(os.sep, "/")
    qc, _, _ = state_circuit()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=11)
    isa = pm.run(qc)
    layout = "/".join(str(x) for x in isa.layout.final_index_layout()) if isa.layout else ""
    print("layout:", layout, "depth", isa.depth(), "epoch", epoch)
    sampler = SamplerV2(mode=backend)

    done = {r["purpose"] for r in ledger_rows() if r["tier"] == "HW1" and r["status"] in ("ok", "failed")}
    for j, rep in plan_order():
        name, S, s, mode, role = j
        key = f"{name}_r{rep}"
        if key in done:
            continue
        p = predicted_usage(S, s)
        used = measured_so_far()
        if used + MARGIN * p > a.cap:
            print(f"STOP before {key}: measured {used:.1f} s + 1.3 x {p:.2f} s would exceed cap {a.cap} s")
            break
        pubs, settings, seed = build_pubs(isa, j, rep)
        tag = f"shadowopt-HW1-{key}-{uuid.uuid4().hex[:12]}"
        rec = {"tag": tag, "intent_utc": now().isoformat(), "job_id": None, "layout": layout, "calibration": cal_rel,
               "epoch": epoch, "settings_seed": seed, "predicted": p}
        pending[key] = rec
        save_pending(pending)  # durable intent BEFORE submission
        sampler.options.environment.job_tags = [tag]
        job = sampler.run(pubs, shots=s)
        rec["job_id"] = job.job_id()
        save_pending(pending)
        print(f"{key}: submitted {job.job_id()} tag {tag} (pred {p:.2f} s)", flush=True)
        if not finish(job, key, rec, pending):
            sys.exit(f"{key}: job {job.job_id()} unresolved; rerun later to reconcile (nothing new submitted)")
    u1 = service.usage()
    json.dump(u1, open(os.path.join(QPU, f"service_usage_{now().strftime('%Y%m%dT%H%M%SZ')}_HW1_after.json"), "w"),
              default=str, indent=1)
    print("usage after:", {k: u1.get(k) for k in ("usage_consumed_seconds", "usage_limit_seconds", "usage_remaining_seconds")})
    print(f"HW1 measured total: {measured_so_far():.1f} s (cap {a.cap})")


def load_pending():
    return json.load(open(PENDING)) if os.path.exists(PENDING) else {}


def save_pending(pending):
    tmp = PENDING + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pending, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PENDING)


def final_usage(job, tries=60, wait=10):
    """Billed QPU seconds once IBM marks billing final (job.metrics()["usage"]["status"] != "pending"; runtime 0.49's
    job.usage() returns 0 while pending). A final 0 is accepted (e.g. a job cancelled before running). Returns None if
    billing is still pending after `tries` polls - the caller keeps the job pending."""
    for _ in range(tries):
        try:
            u = job.metrics().get("usage", {})
        except Exception:
            u = {}
        if u and u.get("status", "pending") != "pending":
            return float(u.get("qpu_charge_time_seconds") or 0.0)
        time.sleep(wait)
    return None


def finish(job, key, rec, pending):
    """Wait for a terminal state, fetch the result (retrying downloads), wait for finalized billing, write the ledger row
    with the SUBMISSION-TIME provenance of `rec`, then drop the pending record. Returns False if unresolved."""
    name, rep = key.rsplit("_r", 1)
    rep = int(rep)
    _, S, s, mode, role = next(g for g in GRID if g[0] == name)
    while True:
        st = str(job.status()).split(".")[-1].upper()
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(15)
    result, notes = None, ""
    if st == "DONE":
        for attempt in range(5):
            try:
                result = job.result()
                break
            except Exception as e:
                notes = f"result download retry {attempt}: {type(e).__name__}"
                time.sleep(30)
        if result is None:
            return False  # paid job, result not yet retrievable: keep pending, never resubmit
    uq = final_usage(job)
    if uq is None:  # billing not final for ANY terminal state: keep pending, never write a row, never submit more
        return False
    status = "ok" if st == "DONE" else "failed"
    settings, seed = job_settings(name, rep, S)
    assert seed == rec["settings_seed"]
    mu, q, run = metrics_times(job)
    dsum, wall = spans(result) if result is not None else (None, None)
    raw = save_raw(job.job_id(), name, rep, settings, result, mode) if result is not None else ""
    append_row({"date_utc": now().isoformat(), "job_id": job.job_id(), "tier": "HW1", "purpose": key, "role": role,
                "mode": mode, "settings": S, "shots": s, "n_exec": S * s, "n_pubs": S if mode == "circ" else 1,
                "predicted_U_s": round(rec["predicted"], 4), "usage_quantum_s": uq, "metrics_usage": mu,
                "span_sum_s": dsum, "span_wall_s": wall, "queue_s": q, "running_s": run, "settings_seed": seed,
                "layout": rec["layout"], "calibration_snapshot": rec["calibration"], "raw_path": raw, "status": status,
                "notes": f"epoch={rec['epoch']}; tag={rec['tag']}; job_status={st}; {notes}"})
    pending.pop(key, None)
    save_pending(pending)
    print(f"{key}: usage {uq} s (pred {rec['predicted']:.2f}); spans {dsum}; queue {q}; status {status}", flush=True)
    return True


if __name__ == "__main__":
    main()
