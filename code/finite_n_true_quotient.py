#!/usr/bin/env python3
"""Recompute finite-N cycle errors with the exact global-phase quotient metric.

The original diagnostic used Euclidean distance in the gauge-fixed complex
coordinate w=Z2 exp(-i arg Z1).  That distance is global-phase invariant, but
it is not the finite chordal quotient distance induced by the physical norm
on (Z1,Z2).  For |Z1|=1 the exact pairwise quotient distance is

 d_Q(w,v)^2 = 2 + |w|^2 + |v|^2 - 2 |1 + conj(w) v|.

We minimize this expression exhaustively over all 8000 points of a dense
OA-cycle reference.  Chunking bounds memory without changing the discrete
minimum.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
REPORTS = BASE / "reports"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE / "build" / "matplotlib"))


def oa_rhs(_t: float, x: np.ndarray, A: float = 0.33, beta: float = 0.10) -> np.ndarray:
    r, psi = x
    mu, nu = (1.0 + A) / 2.0, (1.0 - A) / 2.0
    return np.array(
        [
            (1.0 - r * r) / 2.0
            * (mu * r * np.sin(beta) + nu * np.sin(psi + beta)),
            -mu * np.cos(beta)
            - nu * r * np.cos(beta - psi)
            + (1.0 + r * r)
            / (2.0 * r)
            * (mu * r * np.cos(beta) + nu * np.cos(psi + beta)),
        ]
    )


def oa_cycle(A: float = 0.33, beta: float = 0.10, npts: int = 8000):
    sol = solve_ivp(
        oa_rhs,
        (0.0, 3000.0),
        [0.65, -0.2],
        args=(A, beta),
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=0.1,
        dense_output=True,
    )
    ts = np.arange(2000.0, 3000.0, 0.1)
    dr = np.asarray([oa_rhs(t, sol.sol(t), A, beta)[0] for t in ts])
    peaks = []
    for ta, tb, fa, fb in zip(ts[:-1], ts[1:], dr[:-1], dr[1:]):
        if fa > 0.0 and fb <= 0.0:
            peaks.append(
                brentq(
                    lambda t: oa_rhs(t, sol.sol(t), A, beta)[0],
                    ta,
                    tb,
                    xtol=1.0e-13,
                )
            )
    if len(peaks) < 3:
        raise RuntimeError("No converged OA cycle")
    t0, t1 = peaks[-2], peaks[-1]
    tt = np.linspace(t0, t1, npts, endpoint=False)
    cycle = sol.sol(tt)
    cycle[1] = (cycle[1] + np.pi) % (2.0 * np.pi) - np.pi
    w = cycle[0] * np.exp(-1j * cycle[1])
    embedding = np.column_stack((w.real, w.imag))
    return t1 - t0, cycle, w, embedding


def wrapped_cauchy(rho: float, phi: float, N: int, rng: np.random.Generator):
    u = rng.uniform(-np.pi, np.pi, N)
    return phi + 2.0 * np.arctan((1.0 - rho) / (1.0 + rho) * np.tan(u / 2.0))


def micro_rhs(y: np.ndarray, A: float, beta: float) -> np.ndarray:
    phi1, theta2 = y[0], y[1:]
    mu, nu = (1.0 + A) / 2.0, (1.0 - A) / 2.0
    alpha = np.pi / 2.0 - beta
    z1 = np.exp(1j * phi1)
    z2 = np.mean(np.exp(1j * theta2))
    H1, H2 = mu * z1 + nu * z2, nu * z1 + mu * z2
    return np.r_[
        np.imag(np.exp(-1j * (phi1 + alpha)) * H1),
        np.imag(np.exp(-1j * (theta2 + alpha)) * H2),
    ]


def rk4(y: np.ndarray, dt: float, A: float, beta: float) -> np.ndarray:
    k1 = micro_rhs(y, A, beta)
    k2 = micro_rhs(y + dt * k1 / 2.0, A, beta)
    k3 = micro_rhs(y + dt * k2 / 2.0, A, beta)
    k4 = micro_rhs(y + dt * k3, A, beta)
    return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def quotient_distance_squared(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Broadcasting implementation of the exact chordal quotient distance."""
    d2 = (
        2.0
        + np.abs(w) ** 2
        + np.abs(v) ** 2
        - 2.0 * np.abs(1.0 + np.conj(w) * v)
    )
    return np.maximum(d2, 0.0)


def distances_to_cycle(
    w: np.ndarray,
    cycle_w: np.ndarray,
    cycle_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean and true quotient distances to the sampled cycle."""
    embedding = np.column_stack((w.real, w.imag))
    tree = cKDTree(cycle_embedding)
    euclidean_nearest = tree.query(embedding, k=1)[0]
    true_distance = np.empty(w.size)
    for start in range(0, w.size, 128):
        stop = min(start + 128, w.size)
        d2 = quotient_distance_squared(w[start:stop, None], cycle_w[None, :])
        true_distance[start:stop] = np.sqrt(np.min(d2, axis=1))
    return euclidean_nearest, true_distance


def validate_quotient_formula(
    cases: int = 64,
    tolerance: float = 5.0e-11,
) -> dict[str, float | int | str]:
    """Check the closed form against numerical global-phase minimization.

    This is an independent deterministic unit check of the pairwise quotient
    formula.  It replaces the former per-trajectory ``validation_*`` fields,
    which were hard-coded to zero and therefore did not represent a test.
    """
    rng = np.random.default_rng(20260727)
    w = np.sqrt(rng.uniform(size=cases)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=cases)
    )
    v = np.sqrt(rng.uniform(size=cases)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, size=cases)
    )
    analytic = quotient_distance_squared(w, v)
    numerical = np.empty(cases)
    grid = np.linspace(-np.pi, np.pi, 2049, endpoint=False)
    grid_spacing = 2.0 * np.pi / grid.size

    for index, (wi, vi) in enumerate(zip(w, v)):
        def objective(phase: float) -> float:
            rotation = np.exp(1j * phase)
            return float(
                abs(1.0 - rotation) ** 2
                + abs(wi - rotation * vi) ** 2
            )

        values = np.asarray([objective(phase) for phase in grid])
        center = float(grid[int(np.argmin(values))])
        optimum = minimize_scalar(
            objective,
            bounds=(center - grid_spacing, center + grid_spacing),
            method="bounded",
            options={"xatol": 1.0e-14},
        )
        if not optimum.success:
            raise RuntimeError("global-phase validation minimization failed")
        numerical[index] = float(optimum.fun)

    gaps = np.abs(analytic - numerical)
    failures = int(np.count_nonzero(gaps > tolerance))
    if failures:
        raise RuntimeError(
            f"quotient formula validation failed in {failures}/{cases} cases; "
            f"maximum squared-distance gap={np.max(gaps):.3e}"
        )
    return {
        "method": (
            "closed form compared with deterministic numerical minimization "
            "over the global phase for random pairs in the unit disk"
        ),
        "random_seed": 20260727,
        "number_of_cases": cases,
        "squared_distance_tolerance": tolerance,
        "maximum_absolute_squared_distance_gap": float(np.max(gaps)),
        "failures": failures,
    }


def one_job(job):
    (
        N,
        seed,
        A,
        beta,
        dt,
        T,
        discard,
        r0,
        psi0,
        cycle_w,
        cycle_embedding,
    ) = job
    rng = np.random.default_rng(seed)
    y = np.r_[psi0, wrapped_cauchy(r0, 0.0, N, rng)]
    stride = max(1, int(round(0.2 / dt)))
    out = []
    for k in range(int(round(T / dt))):
        y = rk4(y, dt, A, beta)
        if (k + 1) % stride == 0 and (k + 1) * dt >= discard:
            z2 = np.mean(np.exp(1j * y[1:]))
            r = abs(z2)
            psi = (y[0] - np.angle(z2) + np.pi) % (2.0 * np.pi) - np.pi
            out.append((r, psi))
    out = np.asarray(out)
    w = out[:, 0] * np.exp(-1j * out[:, 1])
    de, dq = distances_to_cycle(w, cycle_w, cycle_embedding)
    return {
        "N": N,
        "seed": seed,
        "dt": dt,
        "mean_r2": float(out[:, 0].mean()),
        "std_r2": float(out[:, 0].std()),
        "rms_euclidean_gauge_distance": float(np.sqrt(np.mean(de * de))),
        "q95_euclidean_gauge_distance": float(np.quantile(de, 0.95)),
        "rms_true_chordal_quotient_distance": float(np.sqrt(np.mean(dq * dq))),
        "q95_true_chordal_quotient_distance": float(np.quantile(dq, 0.95)),
        "quotient_search_cycle_points": int(cycle_w.size),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute(dt: float, seeds: int, workers: int, T: float, discard: float) -> Path:
    A, beta = 0.33, 0.10
    period, cycle, cycle_w, cycle_embedding = oa_cycle(A, beta)
    r0, psi0 = cycle[:, 0]
    sizes = [64, 128, 256, 512, 1024, 2048]
    jobs = [
        (
            N,
            10_000 + 100 * N + s,
            A,
            beta,
            dt,
            T,
            discard,
            r0,
            psi0,
            cycle_w,
            cycle_embedding,
        )
        for N in sizes
        for s in range(seeds)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one_job, jobs))
    suffix = "dt004" if abs(dt - 0.04) < 1.0e-12 else "dt002"
    path = DATA / f"finite_n_raw_true_quotient_{suffix}.csv"
    write_rows(path, rows)
    print(f"OA period={period:.12g}; wrote {path}")
    for N in sizes:
        q = [r for r in rows if r["N"] == N]
        print(
            N,
            "Euclidean=",
            np.mean([r["rms_euclidean_gauge_distance"] for r in q]),
            "quotient=",
            np.mean([r["rms_true_chordal_quotient_distance"] for r in q]),
        )
    return path


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def fit_one(rows: list[dict[str, float]], bootstrap: int, rng) -> dict:
    sizes = np.unique([int(row["N"]) for row in rows])
    metrics = {
        "true_chordal_quotient": "rms_true_chordal_quotient_distance",
        "euclidean_gauge": "rms_euclidean_gauge_distance",
    }
    result: dict[str, object] = {"N": sizes.tolist()}
    for label, field in metrics.items():
        groups = {
            N: np.asarray([row[field] for row in rows if int(row["N"]) == N])
            for N in sizes
        }
        means = np.asarray([np.mean(groups[N]) for N in sizes])
        slope, intercept = np.polyfit(np.log(sizes), np.log(means), 1)
        boot = np.empty(bootstrap)
        for b in range(bootstrap):
            sample_means = [
                np.mean(rng.choice(groups[N], size=groups[N].size, replace=True))
                for N in sizes
            ]
            boot[b] = np.polyfit(np.log(sizes), np.log(sample_means), 1)[0]
        result[label] = {
            "means": means.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
            "bootstrap_95_percent_CI": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
        }
    return result


def summarize(bootstrap: int = 10000) -> None:
    rng = np.random.default_rng(20260716)
    paths = {
        "dt_0p04": DATA / "finite_n_raw_true_quotient_dt004.csv",
        "dt_0p02": DATA / "finite_n_raw_true_quotient_dt002.csv",
    }
    summary = {
        label: fit_one(read_rows(path), bootstrap, rng)
        for label, path in paths.items()
    }
    summary["metric_formula"] = (
        "d_Q(w,v)^2=2+|w|^2+|v|^2-2|1+conj(w)v|, minimized over the OA cycle"
    )
    summary["interpretation"] = (
        "The gauge-fixed Euclidean distance is global-phase invariant but is not "
        "the chordal quotient metric; both are reported for a paired comparison."
    )
    summary["quotient_formula_validation"] = validate_quotient_formula()
    (DATA / "finite_n_true_quotient_fit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained")
    for ax, (label, fit) in zip(axes, summary.items()):
        if not label.startswith("dt_"):
            continue
        N = np.asarray(fit["N"])
        for metric, marker, color in (
            ("euclidean_gauge", "o", "#777777"),
            ("true_chordal_quotient", "s", "#1764ab"),
        ):
            values = np.asarray(fit[metric]["means"])
            slope = fit[metric]["slope"]
            ax.loglog(N, values, marker + "-", color=color,
                      label=f"{metric.replace('_', ' ')}, slope={slope:.3f}")
        ax.set(xlabel="$N$", ylabel="mean RMS cycle distance", title=label.replace("_", "="))
        ax.grid(which="both", alpha=0.22)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(REPORTS / "finite_n_metric_comparison.pdf", bbox_inches="tight")
    fig.savefig(REPORTS / "finite_n_metric_comparison.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--T", type=float, default=800.0)
    parser.add_argument("--discard", type=float, default=250.0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(args.bootstrap)
    else:
        compute(args.dt, args.seeds, args.workers, args.T, args.discard)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
