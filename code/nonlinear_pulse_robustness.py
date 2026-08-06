#!/usr/bin/env python3
"""Nonlinear and practical robustness audit of relaxation across the EP.

The script keeps the original node/EP/focus parameter choices, sweeps signed
coherence kicks over five decades, tests small A and beta offsets, and performs
a reproducible observation-noise template-identification experiment.  It does
not add stochastic forcing to the OA vector field, whose physical meaning
would require a separate microscopic noise model.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))

from local_analysis import benchmark_points  # noqa: E402
from model import two_pop_equilibrium, two_pop_jacobian, two_pop_rhs  # noqa: E402

RESULTS = BASE / "data"
FIGURES = BASE / "reports"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.labelpad": 1.7,
        "xtick.major.pad": 1.7,
        "ytick.major.pad": 1.7,
        "axes.linewidth": 0.70,
        "lines.linewidth": 1.05,
        "legend.handlelength": 1.30,
        "legend.handletextpad": 0.32,
        "legend.borderaxespad": 0.25,
        "legend.borderpad": 0.28,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax, text: str, *, x: float = -0.16) -> None:
    """Place a panel letter outside the plotting area without clipping."""
    ax.text(
        x,
        1.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize=9.0,
        zorder=20,
        clip_on=False,
    )


def save_figure(fig: plt.Figure) -> None:
    """Atomically replace the vector and raster supplementary figure files."""
    pdf = FIGURES / "nonlinear_pulse_robustness.pdf"
    png = FIGURES / "nonlinear_pulse_robustness.png"
    token = uuid.uuid4().hex
    pdf_tmp = FIGURES / f".nonlinear_pulse_robustness.{token}.pdf.tmp"
    png_tmp = FIGURES / f".nonlinear_pulse_robustness.{token}.png.tmp"
    try:
        fig.savefig(pdf_tmp, format="pdf")
        if not pdf_tmp.read_bytes().rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"Incomplete PDF render: {pdf_tmp}")
        fig.savefig(png_tmp, format="png", dpi=300)
        for temporary, final in ((pdf_tmp, pdf), (png_tmp, png)):
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, final)
    finally:
        pdf_tmp.unlink(missing_ok=True)
        png_tmp.unlink(missing_ok=True)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spectral_class(J: np.ndarray, ep_override: bool = False) -> str:
    D = float(np.trace(J) ** 2 - 4.0 * np.linalg.det(J))
    if ep_override or abs(D) < 1.0e-10:
        return "exceptional_point"
    return "stable_node" if D > 0.0 else "stable_focus"


def integrate_response(
    A: float,
    beta: float,
    xeq: np.ndarray,
    epsilon: float,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    perturbation = np.array([epsilon, 0.0])
    if not (0.0 < xeq[0] + epsilon <= 1.0):
        raise ValueError("pulse moves r outside its physical interval")
    sol = solve_ivp(
        lambda t, x: two_pop_rhs(t, x, A, beta),
        (float(time[0]), float(time[-1])),
        xeq + perturbation,
        t_eval=time,
        method="DOP853",
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step=0.20,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    nonlinear = (sol.y.T - xeq) / epsilon
    J = two_pop_jacobian(xeq, A, beta)
    linear = np.asarray([expm(J * t) @ np.array([1.0, 0.0]) for t in time])
    return nonlinear, linear


def zero_crossings(y: np.ndarray, tolerance: float = 1.0e-5) -> int:
    signs = np.sign(y)
    signs[abs(y) < tolerance] = 0.0
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def nominal_cases():
    events = benchmark_points(0.10)
    gap = float(events["EP"]["A"] - events["SN"]["A"])
    Aep = float(events["EP"]["A"])
    return events, [
        ("stable_node", Aep - 0.45 * gap),
        ("exceptional_point", Aep),
        ("stable_focus", Aep + 12.0 * gap),
    ]


def amplitude_sweep(time: np.ndarray):
    events, cases = nominal_cases()
    magnitudes = np.array(
        [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
         1e-3, 3e-3, 1e-2, 3e-2, 5e-2]
    )
    amplitudes = np.r_[-magnitudes[::-1], magnitudes]
    rows: list[dict] = []
    representative: list[dict] = []
    clean_templates: dict[str, np.ndarray] = {}
    equilibria: dict[str, np.ndarray] = {}
    for label, A in cases:
        if label == "exceptional_point":
            xeq = np.array([events["EP"]["r"], events["EP"]["psi"]], float)
        else:
            xeq = two_pop_equilibrium(A, 0.10, (events["EP"]["r"], events["EP"]["psi"]))
        equilibria[label] = xeq
        J = two_pop_jacobian(xeq, A, 0.10)
        eig = np.linalg.eigvals(J)
        for epsilon in amplitudes:
            nonlinear, linear = integrate_response(A, 0.10, xeq, float(epsilon), time)
            diff = nonlinear - linear
            rel_l2_r = float(
                np.linalg.norm(diff[:, 0]) / max(np.linalg.norm(linear[:, 0]), 1.0e-300)
            )
            rel_l2_state = float(
                np.linalg.norm(diff) / max(np.linalg.norm(linear), 1.0e-300)
            )
            final_unscaled = abs(epsilon) * np.linalg.norm(nonlinear[-1])
            rows.append(
                {
                    "case": label,
                    "A": A,
                    "beta": 0.10,
                    "epsilon": epsilon,
                    "spectral_class": spectral_class(J, label == "exceptional_point"),
                    "spectral_abscissa": float(np.max(eig.real)),
                    "relative_L2_error_delta_r": rel_l2_r,
                    "relative_L2_error_full_state": rel_l2_state,
                    "max_abs_error_delta_r_over_epsilon": float(np.max(abs(diff[:, 0]))),
                    "max_abs_nonlinear_delta_r_over_epsilon": float(
                        np.max(abs(nonlinear[:, 0]))
                    ),
                    "time_of_max_abs_delta_r": float(time[np.argmax(abs(nonlinear[:, 0]))]),
                    "delta_r_zero_crossings": zero_crossings(nonlinear[:, 0]),
                    "final_physical_displacement": final_unscaled,
                    "returned_to_local_neighborhood_by_tend": int(final_unscaled < 1.0e-3),
                    "left_local_basin_by_tend": int(final_unscaled > 5.0e-2),
                }
            )
            if np.isclose(abs(epsilon), 1.0e-4) or np.isclose(abs(epsilon), 1.0e-2):
                for k, t in enumerate(time):
                    representative.append(
                        {
                            "case": label,
                            "A": A,
                            "epsilon": epsilon,
                            "time": t,
                            "nonlinear_delta_r_over_epsilon": nonlinear[k, 0],
                            "linear_delta_r_over_epsilon": linear[k, 0],
                            "nonlinear_delta_psi_over_epsilon": nonlinear[k, 1],
                            "linear_delta_psi_over_epsilon": linear[k, 1],
                        }
                    )
            if np.isclose(epsilon, 1.0e-3):
                clean_templates[label] = nonlinear[:, 0].copy()
    return events, cases, equilibria, rows, representative, clean_templates


def parameter_offsets(
    events: dict,
    cases: list[tuple[str, float]],
    equilibria: dict[str, np.ndarray],
    time: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    epsilon = 1.0e-3
    for label, A0 in cases:
        x0 = equilibria[label]
        nominal, _ = integrate_response(A0, 0.10, x0, epsilon, time)
        offsets = [("A", dA, 0.0) for dA in (-1e-5, -5e-6, 0.0, 5e-6, 1e-5)]
        offsets += [
            ("beta", 0.0, db) for db in (-5e-4, -1e-4, 1e-4, 5e-4)
        ]
        for kind, dA, db in offsets:
            A, beta = A0 + dA, 0.10 + db
            try:
                xeq = two_pop_equilibrium(A, beta, x0)
            except RuntimeError:
                rows.append(
                    {
                        "nominal_case": label,
                        "offset_kind": kind,
                        "delta_A": dA,
                        "delta_beta": db,
                        "A": A,
                        "beta": beta,
                        "equilibrium_exists": 0,
                        "r_star": np.nan,
                        "psi_star": np.nan,
                        "discriminant": np.nan,
                        "spectral_class": "no_locked_equilibrium_found",
                        "spectral_abscissa": np.nan,
                        "response_relative_L2_change": np.nan,
                        "response_max_abs_change": np.nan,
                        "delta_r_zero_crossings": -1,
                    }
                )
                continue
            J = two_pop_jacobian(xeq, A, beta)
            D = float(np.trace(J) ** 2 - 4.0 * np.linalg.det(J))
            eig = np.linalg.eigvals(J)
            response, _ = integrate_response(A, beta, xeq, epsilon, time)
            deviation = response[:, 0] - nominal[:, 0]
            rows.append(
                {
                    "nominal_case": label,
                    "offset_kind": kind,
                    "delta_A": dA,
                    "delta_beta": db,
                    "A": A,
                    "beta": beta,
                    "equilibrium_exists": 1,
                    "r_star": xeq[0],
                    "psi_star": xeq[1],
                    "discriminant": D,
                    "spectral_class": spectral_class(J),
                    "spectral_abscissa": float(np.max(eig.real)),
                    "response_relative_L2_change": float(
                        np.linalg.norm(deviation)
                        / max(np.linalg.norm(nominal[:, 0]), 1.0e-300)
                    ),
                    "response_max_abs_change": float(np.max(abs(deviation))),
                    "delta_r_zero_crossings": zero_crossings(response[:, 0]),
                }
            )
    return rows


def observation_noise_test(
    clean_templates: dict[str, np.ndarray],
    time: np.ndarray,
    n_trials: int = 1000,
) -> list[dict]:
    """Identify the three clean relaxation templates after measurement noise."""
    rng = np.random.default_rng(20260716)
    labels = list(clean_templates)
    # One sample every two time units keeps the experiment realistic and avoids
    # treating the ODE integrator's dense grid as independent measurements.
    indices = np.arange(0, time.size, max(1, int(round(2.0 / (time[1] - time[0])))))
    templates = np.asarray([clean_templates[label][indices] for label in labels])
    rows: list[dict] = []
    for sigma_over_epsilon in (0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.00):
        confusion = np.zeros((len(labels), len(labels)), int)
        for i, clean in enumerate(templates):
            noisy = clean + sigma_over_epsilon * rng.standard_normal(
                (n_trials, clean.size)
            )
            sse = np.sum((noisy[:, None, :] - templates[None, :, :]) ** 2, axis=2)
            selected = np.argmin(sse, axis=1)
            for j in range(len(labels)):
                confusion[i, j] = int(np.sum(selected == j))
        for i, truth in enumerate(labels):
            for j, selected in enumerate(labels):
                rows.append(
                    {
                        "sigma_observation_over_pulse": sigma_over_epsilon,
                        "true_case": truth,
                        "selected_case": selected,
                        "count": int(confusion[i, j]),
                        "fraction": float(confusion[i, j] / n_trials),
                        "n_trials_per_true_case": n_trials,
                        "sampling_interval": float(time[indices[1]] - time[indices[0]]),
                        "observation_window": float(time[-1]),
                        "rng_seed": 20260716,
                    }
                )
    return rows


def make_plot(
    sweep: list[dict],
    representative: list[dict],
    offsets: list[dict],
    noise: list[dict],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "stable_node": "#0072B2",
        "exceptional_point": "#009E73",
        "stable_focus": "#D55E00",
    }
    display_names = {
        "stable_node": "node",
        "exceptional_point": "EP",
        "stable_focus": "focus",
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 5.05), layout="constrained")
    fig.get_layout_engine().set(
        w_pad=0.04, h_pad=0.04, wspace=0.16, hspace=0.10
    )

    ax = axes[0, 0]
    for case in colors:
        q = [r for r in representative if r["case"] == case and np.isclose(r["epsilon"], 1e-4)
             and r["epsilon"] > 0.0]
        ax.plot([r["time"] for r in q], [r["nonlinear_delta_r_over_epsilon"] for r in q],
                color=colors[case], label=display_names[case])
        ax.plot([r["time"] for r in q], [r["linear_delta_r_over_epsilon"] for r in q],
                color=colors[case], ls="--", alpha=0.55)
    ax.set(xlabel="$t$", ylabel=r"$\delta r/\epsilon$", title=r"nonlinear (solid), linear (dashed), $\epsilon=10^{-4}$")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper right")
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    for case in colors:
        q = [r for r in sweep if r["case"] == case and r["epsilon"] > 0]
        ax.loglog([r["epsilon"] for r in q], [r["relative_L2_error_delta_r"] for r in q],
                  "o-", ms=3, color=colors[case], label=display_names[case])
    ax.axvline(1e-3, color="black", lw=0.6, ls=":")
    ax.set(xlabel=r"pulse amplitude $\epsilon$", ylabel="relative nonlinear error")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper left")
    panel_label(ax, "(b)", x=-0.20)

    ax = axes[1, 0]
    for case in colors:
        q = [r for r in offsets if r["nominal_case"] == case and r["offset_kind"] == "A"]
        ax.plot([r["delta_A"] for r in q], [r["response_relative_L2_change"] for r in q],
                "o-", ms=3, color=colors[case], label=display_names[case])
    ax.set(xlabel=r"parameter offset $\delta A$", ylabel="relative response change")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper center")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    for case in colors:
        levels = sorted(set(r["sigma_observation_over_pulse"] for r in noise))
        accuracy = []
        for level in levels:
            q = [r for r in noise if r["true_case"] == case and r["selected_case"] == case
                 and r["sigma_observation_over_pulse"] == level]
            accuracy.append(q[0]["fraction"])
        ax.semilogx(levels, accuracy, "o-", ms=3, color=colors[case],
                    label=display_names[case])
    ax.set(xlabel=r"observation noise $\sigma/|\epsilon|$", ylabel="template accuracy", ylim=(-0.03, 1.03))
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="lower left")
    panel_label(ax, "(d)", x=-0.20)

    for ax in axes.ravel():
        ax.grid(alpha=0.20, linewidth=0.50)
    save_figure(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, 600.0, 2401)
    events, cases, equilibria, sweep, representative, templates = amplitude_sweep(time)
    offsets = parameter_offsets(events, cases, equilibria, time)
    noise = observation_noise_test(templates, time)
    write_csv(RESULTS / "nonlinear_pulse_amplitude_sweep.csv", sweep)
    write_csv(RESULTS / "nonlinear_pulse_representative_timeseries.csv", representative)
    write_csv(RESULTS / "pulse_parameter_offset_robustness.csv", offsets)
    write_csv(RESULTS / "pulse_observation_noise_confusion.csv", noise)
    make_plot(sweep, representative, offsets, noise)

    summary = {
        "nominal_A": {label: A for label, A in cases},
        "EP_minus_SN_gap": float(events["EP"]["A"] - events["SN"]["A"]),
        "maximum_relative_L2_error_for_abs_epsilon_le_1e-3": {
            case: max(
                row["relative_L2_error_delta_r"]
                for row in sweep
                if row["case"] == case and abs(row["epsilon"]) <= 1.0e-3
            )
            for case, _ in cases
        },
        "largest_two_sided_pulse_remaining_in_local_neighborhood_by_t600": {
            case: max(
                magnitude
                for magnitude in sorted({abs(row["epsilon"]) for row in sweep if row["case"] == case})
                if all(
                    row["returned_to_local_neighborhood_by_tend"]
                    for row in sweep
                    if row["case"] == case and np.isclose(abs(row["epsilon"]), magnitude)
                )
            )
            for case, _ in cases
        },
        "noise_model": (
            "independent Gaussian observation noise added to delta-r samples; "
            "no stochastic forcing was added to the OA dynamics"
        ),
    }
    (RESULTS / "nonlinear_pulse_robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
