#!/usr/bin/env python3
"""Prepare all fast-to-recompute tables used by the revised manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from local_analysis import (
    adaptive_branch,
    benchmark_points,
    first_lyapunov,
    trace_codimension_loci,
)
from modal_projection import continue_ring_modal, locate_ring_ep
from model import two_pop_transverse_rate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def serializable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [[float(z.real), float(z.imag)] for z in value.ravel()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def critical_points_table() -> dict:
    events = benchmark_points()
    rows = []
    for name in ("SN", "EP", "Hopf"):
        p = events[name]
        eig = np.asarray(p["eig"])
        rows.append(
            {
                "event": name,
                "beta": 0.10,
                "A": p["A"],
                "r": p["r"],
                "psi": p["psi"],
                "lambda1_real": eig[0].real,
                "lambda1_imag": eig[0].imag,
                "lambda2_real": eig[1].real,
                "lambda2_imag": eig[1].imag,
                "lambda_transverse": p["lambda_transverse"],
            }
        )
    bt = events["BT"]
    rows.append(
        {
            "event": "BT",
            "beta": bt["beta"],
            "A": bt["A"],
            "r": bt["r"],
            "psi": bt["psi"],
            "lambda1_real": 0.0,
            "lambda1_imag": 0.0,
            "lambda2_real": 0.0,
            "lambda2_imag": 0.0,
            "lambda_transverse": two_pop_transverse_rate(
                np.array([bt["r"], bt["psi"]]), bt["A"], bt["beta"]
            ),
        }
    )
    fields = list(rows[0])
    write_rows(DATA / "critical_points.csv", rows, fields)
    return events


def physical_branch_table() -> None:
    branch, _events = adaptive_branch()
    rows = []
    for p in branch:
        eig = np.asarray(p["eigenvalues"])
        eig = eig[np.argsort(eig.real)]
        rows.append(
            {
                "A": p["A"],
                "r": p["x"][0],
                "psi": p["x"][1],
                "discriminant": p["discriminant"],
                "lambda1_real": eig[0].real,
                "lambda1_imag": eig[0].imag,
                "lambda2_real": eig[1].real,
                "lambda2_imag": eig[1].imag,
                "g_declared_metric": p["g"],
                "gcrit_declared_metric": p["gcrit"],
                "R_B_declared_metric": p["R_B"],
                "bounded_margin": p["bounded_margin"],
                "lambda_transverse": p["lambda_transverse"],
            }
        )
    write_rows(DATA / "locked_branch_declared_metric.csv", rows, list(rows[0]))


def loci_table() -> None:
    betas = np.unique(np.r_[np.linspace(0.02, 0.22, 41), 0.10])
    loci = trace_codimension_loci(betas)
    rows = []
    for i, beta in enumerate(betas):
        sn, ep, hopf = loci["SN"][i], loci["EP"][i], loci["Hopf"][i]
        rows.append(
            {
                "beta": beta,
                "A_SN": sn["A"],
                "A_EP": ep["A"],
                "A_H": hopf["A"],
                "gap_EP_minus_SN": ep["A"] - sn["A"],
                "gap_H_minus_EP": hopf["A"] - ep["A"],
            }
        )
    write_rows(DATA / "codimension_one_loci.csv", rows, list(rows[0]))


def lyapunov_table() -> None:
    rows = [first_lyapunov(float(beta)) for beta in np.linspace(0.02, 0.13, 12)]
    write_rows(DATA / "hopf_first_lyapunov.csv", rows, list(rows[0]))


def ring_tables() -> dict:
    etas = np.linspace(0.12, 0.30, 181)
    ep_summary = {}
    for M in (4, 5):
        records = continue_ring_modal(M, etas)
        rows = []
        for p in records:
            eig = np.asarray(p["eigenvalues"])
            eig = eig[np.argsort(eig.imag)]
            rows.append(
                {
                    "M": M,
                    "eta": p["eta"],
                    "lambda1_real": eig[0].real,
                    "lambda1_imag": eig[0].imag,
                    "lambda2_real": eig[1].real,
                    "lambda2_imag": eig[1].imag,
                    "discriminant": p["discriminant"],
                    "g": p["g"],
                    "gcrit": p["gcrit"],
                    "R_B": p["R_B"],
                    "bounded_margin": p["bounded_margin"],
                    "spectral_gap": p["spectral_gap"],
                    "invariance_residual": p["invariance_residual"],
                    "spectral_abscissa": p["spectral_abscissa"],
                }
            )
        write_rows(DATA / f"ring_M{M}_modal_branch.csv", rows, list(rows[0]))
        ep_summary[f"M{M}"] = locate_ring_ep(M)
    (DATA / "ring_modal_EP_summary.json").write_text(
        json.dumps(serializable(ep_summary), indent=2), encoding="utf-8"
    )
    return ep_summary


def finite_n_summary(bootstrap_samples: int = 10000) -> dict:
    raw04 = np.loadtxt(
        DATA / "finite_n_raw_true_quotient_dt004.csv", delimiter=",", skiprows=1
    )
    raw02 = np.loadtxt(
        DATA / "finite_n_raw_true_quotient_dt002.csv", delimiter=",", skiprows=1
    )
    sizes = np.unique(raw04[:, 0]).astype(int)

    rows = []
    for N in sizes:
        group = raw04[raw04[:, 0] == N]
        rows.append(
            {
                "N": N,
                "n_seeds": len(group),
                "mean_r2": np.mean(group[:, 3]),
                "sem_r2": np.std(group[:, 3], ddof=1) / np.sqrt(len(group)),
                "mean_rms_cycle_distance": np.mean(group[:, 7]),
                "sem_rms_cycle_distance": np.std(group[:, 7], ddof=1) / np.sqrt(len(group)),
                "mean_q95_cycle_distance": np.mean(group[:, 8]),
            }
        )
    write_rows(DATA / "finite_n_summary.csv", rows, list(rows[0]))
    mean_dist = np.asarray([row["mean_rms_cycle_distance"] for row in rows])
    slope04, intercept04 = np.polyfit(np.log(sizes), np.log(mean_dist), 1)
    sizes02 = np.unique(raw02[:, 0]).astype(int)
    means02 = np.asarray(
        [np.mean(raw02[raw02[:, 0] == N, 7]) for N in sizes02]
    )
    slope02, _intercept02 = np.polyfit(np.log(sizes02), np.log(means02), 1)

    rng = np.random.default_rng(20260716)
    boot = np.empty(bootstrap_samples)
    for b in range(bootstrap_samples):
        sampled_means = []
        for N in sizes:
            values = raw04[raw04[:, 0] == N, 7]
            sampled_means.append(np.mean(rng.choice(values, size=values.size, replace=True)))
        boot[b] = np.polyfit(np.log(sizes), np.log(sampled_means), 1)[0]
    summary = {
        "slope_dt_0p04": float(slope04),
        "intercept_dt_0p04": float(intercept04),
        "bootstrap_95_percent_CI": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "slope_dt_0p02": float(slope02),
        "N": sizes.tolist(),
        "seeds_dt_0p04": 8,
        "seeds_dt_0p02": 3,
        "integration_window": [250.0, 800.0],
        "observable": "RMS exact chordal quotient distance to the OA cycle, minimized over global phase and cycle phase",
        "scope": "finite-horizon OA-like Poisson-kernel ensemble; Delta=0",
    }
    (DATA / "finite_n_fit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    events = critical_points_table()
    physical_branch_table()
    loci_table()
    lyapunov_table()
    rings = ring_tables()
    finite = finite_n_summary(args.bootstrap)
    summary = {
        "critical_points": serializable(events),
        "controlled_ring_EP": serializable(rings),
        "finite_N": finite,
    }
    (DATA / "fast_recomputation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Prepared data in {DATA}")


if __name__ == "__main__":
    main()
