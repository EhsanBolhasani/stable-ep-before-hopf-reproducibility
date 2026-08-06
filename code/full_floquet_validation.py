#!/usr/bin/env python3
"""Compute the complete OA Floquet spectrum of the sampled breathers.

For the planar boundary orbit one multiplier is the autonomous (phase)
multiplier 1.  The remaining in-plane multiplier is computed in two
independent ways: from the monodromy matrix and from Liouville's formula,

    m_parallel = exp(integral_0^T tr J(x(t)) dt).

The boundary-transverse multiplier is obtained from its scalar variational
equation.  Thus the table contains all three multipliers of the three-
dimensional two-population OA quotient before restriction to rho_1=1.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))

from model import (  # noqa: E402
    two_pop_equilibrium,
    two_pop_jacobian,
    two_pop_rhs,
    two_pop_transverse_rate,
)

RESULTS = BASE / "data"
FIGURES = BASE / "reports"


@dataclass
class Cycle:
    A: float
    beta: float
    peak: np.ndarray
    period: float
    period_cv: float
    n_maxima: int


def integrate_to_cycle(A: float, beta: float, x0: np.ndarray, duration: float) -> Cycle:
    """Converge to the attracting cycle and return its last r maximum."""

    def maximum(_t: float, x: np.ndarray) -> float:
        return float(two_pop_rhs(0.0, x, A, beta)[0])

    maximum.direction = -1
    maximum.terminal = False
    sol = solve_ivp(
        lambda t, x: two_pop_rhs(t, x, A, beta),
        (0.0, duration),
        np.asarray(x0, float),
        method="DOP853",
        rtol=3.0e-11,
        atol=3.0e-13,
        max_step=0.5,
        dense_output=True,
        events=maximum,
    )
    maxima = sol.t_events[0]
    if maxima.size < 7:
        raise RuntimeError(f"too few maxima at beta={beta:g}, A={A:g}")
    periods = np.diff(maxima[-6:])
    return Cycle(
        A=A,
        beta=beta,
        peak=np.asarray(sol.sol(maxima[-1]), float),
        period=float(periods[-1]),
        period_cv=float(np.std(periods) / np.mean(periods)),
        n_maxima=int(maxima.size),
    )


def settle_cycle(A: float, beta: float, x0: np.ndarray, duration: float) -> Cycle:
    cycle = integrate_to_cycle(A, beta, x0, duration)
    attempts = 0
    while cycle.period_cv > 3.0e-9 and attempts < 3:
        cycle = integrate_to_cycle(
            A,
            beta,
            cycle.peak,
            max(5000.0, 150.0 * cycle.period),
        )
        attempts += 1
    return cycle


def floquet_row(cycle: Cycle) -> dict[str, float | int]:
    """Integrate state, monodromy, divergence, and transverse rate for one period."""
    A, beta, T = cycle.A, cycle.beta, cycle.period
    y0 = np.r_[cycle.peak, np.eye(2).ravel(), 0.0, 0.0]

    def extended_rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:2]
        phi = y[2:6].reshape(2, 2)
        J = two_pop_jacobian(x, A, beta)
        return np.r_[
            two_pop_rhs(t, x, A, beta),
            (J @ phi).ravel(),
            np.trace(J),
            two_pop_transverse_rate(x, A, beta),
        ]

    sol = solve_ivp(
        extended_rhs,
        (0.0, T),
        y0,
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=min(0.10, T / 1000.0),
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    yf = sol.y[:, -1]
    monodromy = yf[2:6].reshape(2, 2)
    multipliers = np.linalg.eigvals(monodromy)
    trivial_index = int(np.argmin(abs(multipliers - 1.0)))
    nontrivial_index = 1 - trivial_index
    m_trivial = float(np.real_if_close(multipliers[trivial_index]).real)
    m_variational = float(np.real_if_close(multipliers[nontrivial_index]).real)
    integral_trace = float(yf[6])
    integral_transverse = float(yf[7])
    m_liouville = float(np.exp(integral_trace))
    m_transverse = float(np.exp(integral_transverse))
    closure_vector = yf[:2] - cycle.peak
    closure_vector[1] = (closure_vector[1] + np.pi) % (2.0 * np.pi) - np.pi
    closure = float(np.linalg.norm(closure_vector, ord=np.inf))
    return {
        "beta": beta,
        "A": A,
        "period": T,
        "period_cv_last_five": cycle.period_cv,
        "cycle_closure_inf": closure,
        "trivial_multiplier_variational": m_trivial,
        "trivial_multiplier_abs_error": abs(m_trivial - 1.0),
        "inplane_exponent": integral_trace / T,
        "inplane_multiplier_liouville": m_liouville,
        "inplane_multiplier_variational": m_variational,
        "inplane_multiplier_relative_mismatch": abs(m_variational - m_liouville)
        / max(abs(m_liouville), 1.0e-300),
        "transverse_exponent": integral_transverse / T,
        "transverse_multiplier": m_transverse,
        "spectral_radius_nontrivial": max(abs(m_liouville), abs(m_transverse)),
        "n_detected_maxima": cycle.n_maxima,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_cycles() -> list[dict]:
    rows: list[dict] = []

    beta = 0.10
    xeq = two_pop_equilibrium(0.28, beta, (0.59, -0.205))
    x = xeq + np.array([0.015, 0.010])
    previous_period = 40.0
    for i, A in enumerate((0.28, 0.30, 0.33)):
        duration = 7000.0 if i == 0 else max(2600.0, 35.0 * previous_period)
        cycle = settle_cycle(A, beta, x, duration)
        x, previous_period = cycle.peak, cycle.period
        row = floquet_row(cycle)
        rows.append(row)
        print(
            f"beta={beta:.2f} A={A:.3f}: "
            f"Lambda_parallel={row['inplane_exponent']:.9g}, "
            f"Lambda_perp={row['transverse_exponent']:.9g}"
        )

    beta = 0.06
    sample_A = np.array(
        [0.320, 0.325, 0.335, 0.350, 0.365, 0.375,
         0.385, 0.390, 0.392, 0.394, 0.395, 0.396]
    )
    xeq = two_pop_equilibrium(0.30, beta, (0.56, -0.13))
    seed = settle_cycle(0.30, beta, xeq + np.array([0.02, 0.01]), 3600.0)
    x, previous_period = seed.peak, seed.period
    for A in sample_A:
        duration = max(2400.0, 24.0 * previous_period)
        cycle = settle_cycle(float(A), beta, x, duration)
        x, previous_period = cycle.peak, cycle.period
        row = floquet_row(cycle)
        rows.append(row)
        print(
            f"beta={beta:.2f} A={A:.3f}: "
            f"Lambda_parallel={row['inplane_exponent']:.9g}, "
            f"Lambda_perp={row['transverse_exponent']:.9g}"
        )
    return rows


def plot(rows: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), layout="constrained")
    colors = {0.10: "#1764ab", 0.06: "#7137a8"}
    for beta in (0.10, 0.06):
        q = [r for r in rows if abs(float(r["beta"]) - beta) < 1.0e-12]
        A = np.array([float(r["A"]) for r in q])
        lp = np.array([float(r["inplane_exponent"]) for r in q])
        lt = np.array([float(r["transverse_exponent"]) for r in q])
        mp = np.array([float(r["inplane_multiplier_liouville"]) for r in q])
        mt = np.array([float(r["transverse_multiplier"]) for r in q])
        c = colors[beta]
        axes[0].plot(A, lp, "o-", color=c, label=fr"in-plane, $\beta={beta:.2f}$")
        axes[0].plot(A, lt, "s--", color=c, alpha=0.72,
                     label=fr"transverse, $\beta={beta:.2f}$")
        axes[1].semilogy(A, mp, "o-", color=c,
                        label=fr"in-plane, $\beta={beta:.2f}$")
        axes[1].semilogy(A, mt, "s--", color=c, alpha=0.72,
                        label=fr"transverse, $\beta={beta:.2f}$")
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set(xlabel="$A$", ylabel="nontrivial Floquet exponent")
    axes[1].axhline(1.0, color="black", lw=0.7)
    axes[1].set(xlabel="$A$", ylabel="nontrivial Floquet multiplier")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIGURES / "full_floquet_stability.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "full_floquet_stability.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = compute_cycles()
    write_csv(RESULTS / "full_floquet_cycles_independent.csv", rows)
    plot(rows)


if __name__ == "__main__":
    main()
