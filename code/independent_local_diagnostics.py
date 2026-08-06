#!/usr/bin/env python3
"""Reproducible local diagnostics for the two-population OA benchmark.

Nothing in the output tables is copied from the manuscript or from an input
CSV.  Critical points, locked branches, pulse responses, periodic orbits,
transverse Floquet exponents, and homoclinic fits are recomputed from the
model equations every time this script is run.

The fixed model inputs are

    mu=(1+A)/2, nu=(1-A)/2, Delta_1=Delta_2=0,
    rho_1=1, x=(r,psi)=(rho_2,phi_1-phi_2).

The Euclidean pulse and transient-gain diagnostics use the explicitly stated
(r,psi) coordinate metric.  The transverse Floquet exponent is metric-free.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import expm, svdvals
from scipy.optimize import curve_fit, root


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_local"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    }
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#666666"


def rhs(_t: float, x: np.ndarray, A: float, beta: float) -> np.ndarray:
    """Invariant-boundary OA equations (II.8)--(II.9)."""
    r, psi = x
    mu, nu = (1.0 + A) / 2.0, (1.0 - A) / 2.0
    sb, cb = np.sin(beta), np.cos(beta)
    h = (1.0 + r * r) / (2.0 * r)
    return np.array(
        [
            (1.0 - r * r)
            * (mu * r * sb + nu * np.sin(psi + beta))
            / 2.0,
            -mu * cb
            - nu * r * np.cos(beta - psi)
            + h * (mu * r * cb + nu * np.cos(psi + beta)),
        ]
    )


def jacobian(x: np.ndarray, A: float, beta: float) -> np.ndarray:
    """Analytic Jacobian in the fixed x=(r,psi) coordinates."""
    r, psi = x
    mu, nu = (1.0 + A) / 2.0, (1.0 - A) / 2.0
    sb, cb = np.sin(beta), np.cos(beta)
    h = (1.0 + r * r) / (2.0 * r)
    hp = (1.0 - r ** -2) / 2.0
    U = mu * r * sb + nu * np.sin(psi + beta)
    V = mu * r * cb + nu * np.cos(psi + beta)
    return np.array(
        [
            [
                -r * U + (1.0 - r * r) * mu * sb / 2.0,
                (1.0 - r * r) * nu * np.cos(psi + beta) / 2.0,
            ],
            [
                -nu * np.cos(beta - psi) + hp * V + h * mu * cb,
                -nu * r * np.sin(beta - psi)
                - h * nu * np.sin(psi + beta),
            ],
        ]
    )


def equilibrium(A: float, beta: float, guess: np.ndarray) -> np.ndarray:
    sol = root(
        lambda x: rhs(0.0, x, A, beta),
        np.asarray(guess, float),
        jac=lambda x: jacobian(x, A, beta),
        tol=1e-12,
    )
    residual = np.linalg.norm(rhs(0.0, sol.x, A, beta), ord=np.inf)
    if residual > 2e-9:
        raise RuntimeError(
            f"equilibrium failed at A={A:.12g}, beta={beta:.12g}: "
            f"residual={residual:.3e}, {sol.message}"
        )
    return sol.x


def critical_point(
    beta: float, kind: str, guess: tuple[float, float, float]
) -> dict:
    """Solve F=0 plus det(J)=0, D=0, or tr(J)=0."""

    def conditions(y):
        x, A = np.asarray(y[:2]), float(y[2])
        J = jacobian(x, A, beta)
        if kind == "SN":
            scalar = np.linalg.det(J)
        elif kind == "EP":
            scalar = np.trace(J) ** 2 - 4.0 * np.linalg.det(J)
        elif kind == "Hopf":
            scalar = np.trace(J)
        else:
            raise ValueError(kind)
        return np.r_[rhs(0.0, x, A, beta), scalar]

    sol = root(conditions, np.asarray(guess, float), tol=1e-12)
    residual = np.linalg.norm(conditions(sol.x), ord=np.inf)
    if residual > 2e-9:
        raise RuntimeError(
            f"{kind} solve failed: residual={residual:.3e}, {sol.message}"
        )
    x, A = sol.x[:2], float(sol.x[2])
    J = jacobian(x, A, beta)
    eig = np.linalg.eigvals(J)
    return {
        "kind": kind,
        "beta": beta,
        "A": A,
        "r": float(x[0]),
        "psi": float(x[1]),
        "J": J,
        "eig": eig,
        "residual_inf": float(residual),
    }


def transverse_instantaneous(x: np.ndarray, A: float, beta: float) -> float:
    """Scalar transverse variational rate at rho_1=1."""
    r, psi = x
    mu, nu = (1.0 + A) / 2.0, (1.0 - A) / 2.0
    return float(
        -(mu * np.sin(beta) + nu * r * np.sin(beta - psi))
    )


def state_row(A: float, beta: float, x: np.ndarray, tag: str) -> dict:
    J = jacobian(x, A, beta)
    eig = np.linalg.eigvals(J)
    if np.max(np.abs(eig.imag)) < 1e-9:
        eig = eig[np.argsort(eig.real)]
    else:
        eig = eig[np.argsort(eig.imag)]
    a, b = J[0]
    c, d = J[1]
    s, g = (b + c) / 2.0, (b - c) / 2.0
    gcrit = np.sqrt(((a - d) ** 2 + 4.0 * s * s) / 4.0)
    return {
        "tag": tag,
        "beta": beta,
        "A": A,
        "r": x[0],
        "psi": x[1],
        "J11": a,
        "J12": b,
        "J21": c,
        "J22": d,
        "trace": np.trace(J),
        "determinant": np.linalg.det(J),
        "discriminant": np.trace(J) ** 2 - 4.0 * np.linalg.det(J),
        "lambda1_real": eig[0].real,
        "lambda1_imag": eig[0].imag,
        "lambda2_real": eig[1].real,
        "lambda2_imag": eig[1].imag,
        "s": s,
        "g": g,
        "gcrit": gcrit,
        "abs_g_over_gcrit": abs(g) / gcrit,
        "lambda_transverse": transverse_instantaneous(x, A, beta),
    }


def write_rows(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_adaptive_locked_branch(beta: float = 0.10):
    """Resolve the very narrow SN--EP interval and continue through Hopf."""
    sn = critical_point(beta, "SN", (0.821, -0.218, 0.17834))
    ep = critical_point(beta, "EP", (0.815, -0.217, 0.17845))
    hopf = critical_point(beta, "Hopf", (0.589, -0.204, 0.2778))
    Asn, Aep, Ah = sn["A"], ep["A"], hopf["A"]
    gap = Aep - Asn
    xep = np.array([ep["r"], ep["psi"]])

    # Continue downward from the EP. q is fractional distance from the SN.
    q_down = np.r_[
        np.linspace(1.0 - 1e-6, 0.08, 74, endpoint=False),
        np.geomspace(0.08, 1e-5, 72),
    ]
    below_desc = []
    guess = xep.copy()
    for q in q_down:
        A = Asn + gap * q
        guess = equilibrium(A, beta, guess)
        below_desc.append(state_row(A, beta, guess, "locked"))

    # Continue upward with logarithmic resolution at the EP and a regular grid
    # over the long stable-focus interval ending at Hopf.
    near = gap * np.geomspace(1e-5, 30.0, 90)
    far_start = Aep + near[-1]
    far = np.linspace(far_start, Ah, 180)
    A_up = np.unique(np.r_[Aep, Aep + near, far])
    above = []
    guess = xep.copy()
    for A in A_up:
        guess = equilibrium(float(A), beta, guess)
        tag = "EP" if abs(A - Aep) < 1e-14 else "locked"
        above.append(state_row(float(A), beta, guess, tag))

    sn_row = state_row(Asn, beta, np.array([sn["r"], sn["psi"]]), "SN")
    hopf_row = state_row(
        Ah, beta, np.array([hopf["r"], hopf["psi"]]), "Hopf"
    )
    rows = [sn_row] + list(reversed(below_desc)) + above
    # Replace the final regular-grid row by the independently solved Hopf row.
    rows[-1] = hopf_row
    write_rows(OUT / "locked_branch_beta_0p10.csv", rows)
    return sn, ep, hopf, rows


def compute_relaxation_pulses(sn: dict, ep: dict, hopf: dict):
    """Small-pulse linear relaxation below, at, and above the EP."""
    beta = ep["beta"]
    gap = ep["A"] - sn["A"]
    cases = [
        ("stable_node", ep["A"] - 0.45 * gap),
        ("exceptional_point", ep["A"]),
        # Far enough above the extremely narrow SN--EP gap to display the
        # damped focus in a finite observation window, yet far below Hopf.
        ("stable_focus", min(ep["A"] + 12.0 * gap, hopf["A"] - gap)),
    ]
    matrices = []
    guess = np.array([ep["r"], ep["psi"]])
    for label, A in cases:
        if label == "exceptional_point":
            x = np.array([ep["r"], ep["psi"]])
        else:
            x = equilibrium(A, beta, guess)
        guess = x
        J = jacobian(x, A, beta)
        matrices.append((label, A, x, J, np.linalg.eigvals(J)))

    decay_windows = []
    for _, _, _, _, eig in matrices:
        alpha = np.max(eig.real)
        decay_windows.append(8.0 / max(-alpha, 1e-4))
        omega = np.max(abs(eig.imag))
        if omega > 1e-6:
            decay_windows.append(3.0 * 2.0 * np.pi / omega)
    t_end = min(1200.0, max(decay_windows))
    times = np.linspace(0.0, t_end, 1801)
    unit_r_pulse = np.array([1.0, 0.0])
    rows, summaries = [], []
    for label, A, x, J, eig in matrices:
        response = np.array([expm(J * t) @ unit_r_pulse for t in times])
        gains = np.array([svdvals(expm(J * t))[0] for t in times])
        numerical_abscissa = np.max(np.linalg.eigvalsh((J + J.T) / 2.0))
        for t, y in zip(times, response):
            rows.append(
                {
                    "case": label,
                    "A": A,
                    "time": t,
                    "delta_r_over_epsilon": y[0],
                    "delta_psi_over_epsilon": y[1],
                }
            )
        summaries.append(
            {
                "case": label,
                "A": A,
                "r_star": x[0],
                "psi_star": x[1],
                "lambda1_real": eig[0].real,
                "lambda1_imag": eig[0].imag,
                "lambda2_real": eig[1].real,
                "lambda2_imag": eig[1].imag,
                "numerical_abscissa": numerical_abscissa,
                "max_transient_gain_fixed_metric": np.max(gains),
                "time_of_max_gain": times[np.argmax(gains)],
            }
        )
    write_rows(OUT / "relaxation_pulses_beta_0p10.csv", rows)
    write_rows(OUT / "relaxation_pulse_summary_beta_0p10.csv", summaries)
    return rows, summaries


@dataclass
class CycleResult:
    A: float
    beta: float
    period: float
    transverse_exponent: float
    transverse_multiplier: float
    inplane_exponent: float
    inplane_multiplier: float
    amplitude_r: float
    period_cv_last: float
    n_maxima: int
    peak_state: np.ndarray


def integrate_cycle(
    A: float,
    beta: float,
    x0: np.ndarray,
    duration: float,
) -> CycleResult:
    """Integrate to a stable cycle and use the last two r maxima."""

    def maximum_event(t, x):
        return rhs(t, x, A, beta)[0]

    maximum_event.direction = -1
    maximum_event.terminal = False
    sol = solve_ivp(
        lambda t, x: rhs(t, x, A, beta),
        (0.0, duration),
        np.asarray(x0, float),
        method="DOP853",
        rtol=3e-11,
        atol=3e-13,
        max_step=0.5,
        dense_output=True,
        events=maximum_event,
    )
    maxima = sol.t_events[0]
    if len(maxima) < 5:
        raise RuntimeError(
            f"too few maxima at A={A}, beta={beta}: {len(maxima)}"
        )
    periods = np.diff(maxima[-6:])
    period = float(periods[-1])
    period_cv = float(np.std(periods) / np.mean(periods))
    t0, t1 = float(maxima[-2]), float(maxima[-1])
    tt = np.linspace(t0, t1, 3001)
    xx = sol.sol(tt)
    instantaneous = np.array(
        [transverse_instantaneous(xx[:, k], A, beta) for k in range(tt.size)]
    )
    divergence = np.array(
        [np.trace(jacobian(xx[:, k], A, beta)) for k in range(tt.size)]
    )
    exponent = float(trapezoid(instantaneous, tt) / period)
    inplane_exponent = float(trapezoid(divergence, tt) / period)
    amplitude = float((np.max(xx[0]) - np.min(xx[0])) / 2.0)
    return CycleResult(
        A=A,
        beta=beta,
        period=period,
        transverse_exponent=exponent,
        transverse_multiplier=float(np.exp(exponent * period)),
        inplane_exponent=inplane_exponent,
        inplane_multiplier=float(np.exp(inplane_exponent * period)),
        amplitude_r=amplitude,
        period_cv_last=period_cv,
        n_maxima=len(maxima),
        peak_state=sol.sol(t1),
    )


def cycle_row(cycle: CycleResult, purpose: str) -> dict:
    return {
        "purpose": purpose,
        "beta": cycle.beta,
        "A": cycle.A,
        "period": cycle.period,
        "transverse_floquet_exponent": cycle.transverse_exponent,
        "transverse_multiplier": cycle.transverse_multiplier,
        "inplane_floquet_exponent": cycle.inplane_exponent,
        "inplane_multiplier": cycle.inplane_multiplier,
        "trivial_multiplier": 1.0,
        "half_peak_to_peak_r": cycle.amplitude_r,
        "period_cv_last_five": cycle.period_cv_last,
        "n_detected_maxima": cycle.n_maxima,
    }


def compute_periodic_orbits():
    """Directly compute all periodic data used by Floquet and global fits."""
    all_rows = []

    # beta=0.10 transverse checks.  The first point is close to Hopf and needs
    # a longer transient because the radial Floquet exponent is small.
    beta = 0.10
    xeq = equilibrium(0.28, beta, np.array([0.59, -0.205]))
    x = xeq + np.array([0.015, 0.01])
    previous_period = 40.0
    for i, A in enumerate((0.28, 0.30, 0.33)):
        duration = 7000.0 if i == 0 else max(2600.0, 35.0 * previous_period)
        cyc = integrate_cycle(A, beta, x, duration)
        # Very close to Hopf the radial attraction is weak. Continue in chunks
        # until the period itself, rather than only the trajectory, is settled.
        retries = 0
        while cyc.period_cv_last > 3e-9 and retries < 3:
            cyc = integrate_cycle(
                A, beta, cyc.peak_state, max(5000.0, 150.0 * cyc.period)
            )
            retries += 1
        x, previous_period = cyc.peak_state, cyc.period
        all_rows.append(cycle_row(cyc, "transverse_check"))

    # beta=0.06 period sequence used for the homoclinic test.  The A values are
    # sampling choices, not copied period data.  Continuation reuses the last
    # maximum as the next initial condition.
    beta = 0.06
    sample_A = np.array(
        [
            0.320,
            0.325,
            0.335,
            0.350,
            0.365,
            0.375,
            0.385,
            0.390,
            0.392,
            0.394,
            0.395,
            0.396,
        ]
    )
    xeq = equilibrium(0.30, beta, np.array([0.56, -0.13]))
    seed = integrate_cycle(
        0.30, beta, xeq + np.array([0.02, 0.01]), 3600.0
    )
    x, previous_period = seed.peak_state, seed.period
    hom_rows = []
    for A in sample_A:
        duration = max(2400.0, 24.0 * previous_period)
        cyc = integrate_cycle(float(A), beta, x, duration)
        # One automatic retry if the last periods are not yet stationary.
        if cyc.period_cv_last > 3e-7:
            cyc = integrate_cycle(
                float(A), beta, cyc.peak_state, max(2400.0, 28.0 * cyc.period)
            )
        x, previous_period = cyc.peak_state, cyc.period
        row = cycle_row(cyc, "homoclinic_period_and_transverse")
        all_rows.append(row)
        hom_rows.append(row)

    write_rows(OUT / "transverse_floquet_cycles.csv", all_rows)
    write_rows(OUT / "homoclinic_periods_beta_0p06.csv", hom_rows)
    return all_rows, hom_rows


def log_period(A, C, c, Ac):
    return C - c * np.log(Ac - A)


def snic_period(A, C, k, Ac):
    return C + k / np.sqrt(Ac - A)


def fit_homoclinic_windows(hom_rows: list[dict]):
    """Re-fit every trailing window and compare with a three-parameter SNIC."""
    Aall = np.array([row["A"] for row in hom_rows])
    Tall = np.array([row["period"] for row in hom_rows])

    def aicc(residual, n_parameters=3):
        n = len(residual)
        sse = np.sum(np.asarray(residual) ** 2)
        return (
            n * np.log(sse / n)
            + 2.0 * n_parameters
            + 2.0
            * n_parameters
            * (n_parameters + 1)
            / (n - n_parameters - 1)
        )

    rows = []
    fits = {}
    for n in range(6, len(Aall) + 1):
        A, T = Aall[-n:], Tall[-n:]
        lower = A.max() + 1e-8
        plog, covlog = curve_fit(
            log_period,
            A,
            T,
            p0=(-40.0, 31.0, lower + 1.6e-4),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.45]),
            maxfev=200000,
        )
        psnic, _ = curve_fit(
            snic_period,
            A,
            T,
            p0=(20.0, 7.0, lower + 1.4e-3),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.45]),
            maxfev=200000,
        )
        res_log = T - log_period(A, *plog)
        res_snic = T - snic_period(A, *psnic)
        Ac = float(plog[2])
        saddle = equilibrium(Ac, 0.06, np.array([0.982, -0.197]))
        saddle_eig = np.linalg.eigvals(jacobian(saddle, Ac, 0.06))
        lam_u, lam_s = np.max(saddle_eig.real), np.min(saddle_eig.real)
        inv_c = 1.0 / plog[1]
        inv_c_se = np.sqrt(np.diag(covlog))[1] / plog[1] ** 2
        rows.append(
            {
                "n_last_points": n,
                "A_min": A.min(),
                "A_max": A.max(),
                "Ac_log": Ac,
                "inv_c_log": inv_c,
                "inv_c_standard_error": inv_c_se,
                "lambda_u_at_Ac": lam_u,
                "lambda_s_at_Ac": lam_s,
                "saddle_quantity": lam_u + lam_s,
                "relative_rate_mismatch": abs(inv_c - lam_u) / lam_u,
                "delta_AICc_log_over_snic": aicc(res_snic) - aicc(res_log),
                "rms_log": np.sqrt(np.mean(res_log ** 2)),
                "rms_snic": np.sqrt(np.mean(res_snic ** 2)),
                "Ac_snic": psnic[2],
            }
        )
        fits[n] = (plog, psnic)
    write_rows(OUT / "homoclinic_fit_windows_beta_0p06.csv", rows)
    return rows, fits


def save_figure(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def plot_branch(rows: list[dict], sn: dict, ep: dict, hopf: dict):
    A = np.array([row["A"] for row in rows])
    D = np.array([row["discriminant"] for row in rows])
    lr1 = np.array([row["lambda1_real"] for row in rows])
    lr2 = np.array([row["lambda2_real"] for row in rows])
    li = np.maximum(
        abs(np.array([row["lambda1_imag"] for row in rows])),
        abs(np.array([row["lambda2_imag"] for row in rows])),
    )
    lp = np.array([row["lambda_transverse"] for row in rows])
    gap = ep["A"] - sn["A"]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    near_mask = A <= ep["A"] + 1.4 * gap
    ax[0].plot(A[near_mask], D[near_mask], color=BLUE)
    ax[0].axhline(0.0, color=GRAY, lw=0.8)
    ax[0].axvline(sn["A"], color=GREEN, ls="--", label="SN")
    ax[0].axvline(ep["A"], color=ORANGE, ls=":", label="EP")
    ax[0].set_xlim(sn["A"] - 0.03 * gap, ep["A"] + 1.4 * gap)
    ax[0].set(xlabel="A", ylabel="discriminant D")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.18)

    ax[1].plot(A, lr1, color=BLUE, label="Re lambda 1")
    ax[1].plot(A, lr2, color=PURPLE, label="Re lambda 2")
    ax[1].plot(A, li, color=ORANGE, ls="--", label="abs Im lambda")
    ax[1].plot(A, lp, color=GRAY, ls=":", label="transverse")
    ax[1].axhline(0.0, color="black", lw=0.7)
    ax[1].axvline(ep["A"], color=ORANGE, ls=":")
    ax[1].axvline(hopf["A"], color=GREEN, ls="--")
    ax[1].set(xlabel="A", ylabel="eigenvalue component")
    ax[1].legend(frameon=False, ncol=2)
    ax[1].grid(alpha=0.18)
    save_figure(fig, "branch_SN_EP_Hopf")


def plot_pulses(pulse_rows: list[dict]):
    fig, ax = plt.subplots(figsize=(4.1, 3.0), constrained_layout=True)
    colors = {
        "stable_node": BLUE,
        "exceptional_point": ORANGE,
        "stable_focus": GREEN,
    }
    labels = {
        "stable_node": "node: two real rates",
        "exceptional_point": "EP: critical decay",
        "stable_focus": "focus: damped oscillation",
    }
    for case in colors:
        subset = [row for row in pulse_rows if row["case"] == case]
        ax.plot(
            [row["time"] for row in subset],
            [row["delta_r_over_epsilon"] for row in subset],
            color=colors[case],
            label=labels[case],
        )
    ax.axhline(0.0, color=GRAY, lw=0.7)
    ax.set(xlabel="time", ylabel="linear response delta r / epsilon")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    save_figure(fig, "relaxation_pulse_across_EP")


def plot_floquet(cycle_rows: list[dict]):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    for beta, color in ((0.10, BLUE), (0.06, ORANGE)):
        sub = [row for row in cycle_rows if abs(row["beta"] - beta) < 1e-12]
        ax[0].plot(
            [row["A"] for row in sub],
            [row["transverse_floquet_exponent"] for row in sub],
            "o-",
            color=color,
            label=f"beta={beta:.2f}",
        )
        ax[1].semilogy(
            [row["A"] for row in sub],
            [row["transverse_multiplier"] for row in sub],
            "o-",
            color=color,
            label=f"beta={beta:.2f}",
        )
    ax[0].axhline(0.0, color=GRAY, lw=0.8)
    ax[0].set(xlabel="A", ylabel="transverse Floquet exponent")
    ax[1].set(xlabel="A", ylabel="transverse multiplier")
    for a in ax:
        a.legend(frameon=False)
        a.grid(alpha=0.18)
    save_figure(fig, "transverse_Floquet_breathers")


def plot_homoclinic(
    hom_rows: list[dict], sensitivity: list[dict], fits: dict
):
    A = np.array([row["A"] for row in hom_rows])
    T = np.array([row["period"] for row in hom_rows])
    nfull = len(A)
    plog, psnic = fits[nfull]
    xx = np.linspace(A.min(), min(A.max() + 5e-5, plog[2] - 1e-6), 500)
    n = np.array([row["n_last_points"] for row in sensitivity])

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.3), constrained_layout=True)
    ax[0, 0].plot(A, T, "o", color=BLUE, label="direct integration")
    ax[0, 0].plot(xx, log_period(xx, *plog), color=ORANGE, label="log fit")
    ax[0, 0].plot(
        xx, snic_period(xx, *psnic), color=GRAY, ls="--", label="SNIC fit"
    )
    ax[0, 0].set(xlabel="A", ylabel="period T")
    ax[0, 0].legend(frameon=False)

    ax[0, 1].plot(
        n, [row["Ac_log"] for row in sensitivity], "o-", color=ORANGE
    )
    ax[0, 1].set(xlabel="number of last points", ylabel="fitted Ac")
    ax[0, 1].ticklabel_format(axis="y", style="plain", useOffset=False)

    ax[1, 0].errorbar(
        n,
        [row["inv_c_log"] for row in sensitivity],
        yerr=[row["inv_c_standard_error"] for row in sensitivity],
        fmt="o-",
        color=BLUE,
        label="inverse log slope",
    )
    ax[1, 0].plot(
        n,
        [row["lambda_u_at_Ac"] for row in sensitivity],
        "s--",
        color=GREEN,
        label="saddle lambda_u",
    )
    ax[1, 0].set(xlabel="number of last points", ylabel="rate")
    ax[1, 0].legend(frameon=False)

    ax[1, 1].plot(
        n,
        [row["delta_AICc_log_over_snic"] for row in sensitivity],
        "o-",
        color=PURPLE,
    )
    ax[1, 1].axhline(0.0, color=GRAY, lw=0.8)
    ax[1, 1].set(xlabel="number of last points", ylabel="Delta AICc (SNIC - log)")
    for a in ax.flat:
        a.grid(alpha=0.18)
    save_figure(fig, "homoclinic_fit_robustness")


def json_safe_critical(point: dict) -> dict:
    return {
        "kind": point["kind"],
        "beta": point["beta"],
        "A": point["A"],
        "r": point["r"],
        "psi": point["psi"],
        "J": point["J"].tolist(),
        "eigenvalues": [
            {"real": float(z.real), "imag": float(z.imag)} for z in point["eig"]
        ],
        "lambda_transverse": transverse_instantaneous(
            np.array([point["r"], point["psi"]]),
            point["A"],
            point["beta"],
        ),
        "residual_inf": point["residual_inf"],
    }


def main():
    sn, ep, hopf, branch_rows = compute_adaptive_locked_branch()
    pulse_rows, pulse_summary = compute_relaxation_pulses(sn, ep, hopf)
    cycle_rows, hom_rows = compute_periodic_orbits()
    sensitivity, fits = fit_homoclinic_windows(hom_rows)

    plot_branch(branch_rows, sn, ep, hopf)
    plot_pulses(pulse_rows)
    plot_floquet(cycle_rows)
    plot_homoclinic(hom_rows, sensitivity, fits)

    summary = {
        "critical_points": {
            point["kind"]: json_safe_critical(point)
            for point in (sn, ep, hopf)
        },
        "SN_to_EP_gap": ep["A"] - sn["A"],
        "EP_to_Hopf_gap": hopf["A"] - ep["A"],
        "pulse_cases": pulse_summary,
        "minimum_transverse_floquet_exponent": min(
            row["transverse_floquet_exponent"] for row in cycle_rows
        ),
        "maximum_transverse_floquet_exponent": max(
            row["transverse_floquet_exponent"] for row in cycle_rows
        ),
        "homoclinic_window_sensitivity": sensitivity,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote reproducible diagnostics to {OUT}")


if __name__ == "__main__":
    main()
