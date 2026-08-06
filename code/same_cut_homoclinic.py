#!/usr/bin/env python3
"""Direct same-cut homoclinic shooting and period scaling at beta=0.10.

Run this script from the repository root as

    python code/same_cut_homoclinic.py --output-dir data

It does not infer the connection from a period fit.  It first matches the
outgoing unstable and incoming stable manifolds of the saddle on a transverse
section, and only then uses that independently located parameter in the
period-scaling check.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq, curve_fit

from model import (
    two_pop_equilibrium,
    two_pop_jacobian,
    two_pop_rhs,
    two_pop_transverse_rate,
)


BETA = 0.10
DEFAULT_BRACKET = (0.3535, 0.3541)
SOLVER_REFINEMENT = (
    (3.0e-9, 3.0e-11, 0.60),
    (3.0e-10, 3.0e-12, 0.30),
    (3.0e-11, 3.0e-13, 0.15),
    (3.0e-12, 3.0e-14, 0.075),
    (1.0e-12, 1.0e-14, 0.05),
)


def saddle_and_vectors(A: float):
    """Return the saddle and consistently oriented unit eigenvectors."""
    saddle = two_pop_equilibrium(A, BETA, (0.945, -0.299))
    eigenvalues, eigenvectors = np.linalg.eig(
        two_pop_jacobian(saddle, A, BETA)
    )
    i_u, i_s = int(np.argmax(eigenvalues.real)), int(np.argmin(eigenvalues.real))
    v_u = eigenvectors[:, i_u].real
    v_s = eigenvectors[:, i_s].real
    v_u /= np.linalg.norm(v_u)
    v_s /= np.linalg.norm(v_s)
    # The homoclinic loop leaves and returns on the lower-r branches.
    if v_u[0] > 0.0:
        v_u *= -1.0
    if v_s[0] > 0.0:
        v_s *= -1.0
    return saddle, eigenvalues.real, v_u, v_s


def manifold_shoot(
    A: float,
    r_section: float = 0.5,
    epsilon: float = 1.0e-7,
    rtol: float = 3.0e-11,
    atol: float = 3.0e-13,
    max_step: float = 0.15,
    dense_output: bool = False,
):
    """Shoot both saddle manifolds to the same transverse section.

    The unstable branch is integrated forward to its upward crossing after
    completing the large excursion.  The stable branch is integrated backward
    to its downward crossing.  At a homoclinic connection these two section
    points coincide.  Since r is fixed, the scalar splitting is Delta psi.
    """
    saddle, eigenvalues, v_u, v_s = saddle_and_vectors(A)

    def upward(_t, x):
        return x[0] - r_section

    upward.direction = 1
    upward.terminal = True

    def downward(_t, x):
        return x[0] - r_section

    downward.direction = -1
    downward.terminal = True

    common = dict(
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
    )
    unstable = solve_ivp(
        lambda t, x: two_pop_rhs(t, x, A, BETA),
        (0.0, 800.0),
        saddle + epsilon * v_u,
        events=upward,
        **common,
    )
    stable_backward = solve_ivp(
        lambda t, x: -two_pop_rhs(t, x, A, BETA),
        (0.0, 800.0),
        saddle + epsilon * v_s,
        events=downward,
        **common,
    )
    if len(unstable.t_events[0]) != 1 or len(stable_backward.t_events[0]) != 1:
        raise RuntimeError(
            f"manifold did not reach r={r_section:g} at A={A:.12g}"
        )
    x_u = unstable.y_events[0][0]
    x_s = stable_backward.y_events[0][0]
    return {
        "A": float(A),
        "saddle": saddle,
        "eigenvalues": eigenvalues,
        "v_u": v_u,
        "v_s": v_s,
        "unstable_section": x_u,
        "stable_section": x_s,
        "splitting_delta_psi": float(x_u[1] - x_s[1]),
        "unstable_flight_time": float(unstable.t_events[0][0]),
        "stable_backward_flight_time": float(stable_backward.t_events[0][0]),
        "unstable_solution": unstable,
        "stable_backward_solution": stable_backward,
    }


def locate_connection(
    r_section: float = 0.5,
    epsilon: float = 1.0e-7,
    bracket: tuple[float, float] = DEFAULT_BRACKET,
    rtol: float = 3.0e-11,
    atol: float = 3.0e-13,
    max_step: float = 0.15,
) -> float:
    """Locate the zero of the manifold splitting for declared IVP settings."""
    return float(
        brentq(
            lambda A: manifold_shoot(
                A,
                r_section,
                epsilon,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )["splitting_delta_psi"],
            bracket[0],
            bracket[1],
            xtol=2.0e-13,
            rtol=1.0e-12,
            maxiter=60,
        )
    )


def solver_refinement_rows(
    r_section: float = 0.5,
    epsilon: float = 1.0e-7,
) -> list[dict[str, float]]:
    """Repeat the connection calculation under IVP tolerance refinement.

    The comparison probes common solver bias that is invisible to a
    section/launch-distance sweep performed at one fixed IVP tolerance.
    The tightest calculation is used only as a numerical reference, not as a
    rigorous enclosure of the true connection parameter.
    """
    rows: list[dict[str, float]] = []
    for rtol, atol, max_step in SOLVER_REFINEMENT:
        connection = locate_connection(
            r_section=r_section,
            epsilon=epsilon,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        rows.append(
            {
                "beta": BETA,
                "r_section": r_section,
                "epsilon": epsilon,
                "ivp_method": "DOP853",
                "ivp_rtol": rtol,
                "ivp_atol": atol,
                "ivp_max_step": max_step,
                "brent_xtol": 2.0e-13,
                "brent_rtol": 1.0e-12,
                "A_connection": connection,
            }
        )
        print(
            "solver refinement: "
            f"rtol={rtol:.1e}, atol={atol:.1e}, "
            f"hmax={max_step:g}, A={connection:.16f}"
        )
    tightest = float(rows[-1]["A_connection"])
    for row in rows:
        row["difference_from_tightest"] = (
            float(row["A_connection"]) - tightest
        )
        row["absolute_difference_from_tightest"] = abs(
            float(row["difference_from_tightest"])
        )
    return rows


@dataclass
class Cycle:
    A: float
    period: float
    period_cv: float
    amplitude_r: float
    transverse_exponent: float
    transverse_multiplier: float
    peak_state: np.ndarray
    n_maxima: int


def integrate_cycle(A: float, x0: np.ndarray, duration: float) -> Cycle:
    """Continue an attracting cycle and measure its last five periods."""

    def maximum(t, x):
        return two_pop_rhs(t, x, A, BETA)[0]

    maximum.direction = -1
    maximum.terminal = False
    sol = solve_ivp(
        lambda t, x: two_pop_rhs(t, x, A, BETA),
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
    if len(maxima) < 6:
        raise RuntimeError(f"only {len(maxima)} maxima at A={A:.12g}")
    periods = np.diff(maxima[-6:])
    t0, t1 = float(maxima[-2]), float(maxima[-1])
    period = t1 - t0
    time = np.linspace(t0, t1, 4001)
    orbit = sol.sol(time)
    transverse = np.array(
        [two_pop_transverse_rate(orbit[:, k], A, BETA) for k in range(time.size)]
    )
    exponent = float(trapezoid(transverse, time) / period)
    return Cycle(
        A=float(A),
        period=float(period),
        period_cv=float(np.std(periods) / np.mean(periods)),
        amplitude_r=float(0.5 * np.ptp(orbit[0])),
        transverse_exponent=exponent,
        transverse_multiplier=float(np.exp(exponent * period)),
        peak_state=np.asarray(sol.sol(t1)),
        n_maxima=len(maxima),
    )


def period_series():
    sample_A = np.array(
        [
            0.300000,
            0.320000,
            0.340000,
            0.345000,
            0.348000,
            0.350000,
            0.351000,
            0.352000,
            0.352500,
            0.353000,
            0.353200,
            0.353400,
            0.353500,
            0.353580,
            0.353630,
            0.353650,
            0.353660,
            0.353680,
            0.353688,
            0.353692,
            0.353694,
            0.353695,
            0.353696,
        ]
    )
    equilibrium = two_pop_equilibrium(0.295, BETA, (0.56, -0.21))
    seed = integrate_cycle(0.295, equilibrium + np.array([0.02, 0.01]), 4000.0)
    state, previous_period = seed.peak_state, seed.period
    rows = []
    for A in sample_A:
        cycle = integrate_cycle(
            float(A), state, max(2800.0, 32.0 * previous_period)
        )
        if cycle.period_cv > 5.0e-8:
            cycle = integrate_cycle(
                float(A), cycle.peak_state, max(5000.0, 40.0 * cycle.period)
            )
        rows.append(
            {
                "beta": BETA,
                "A": cycle.A,
                "period": cycle.period,
                "period_cv_last_five": cycle.period_cv,
                "half_peak_to_peak_r": cycle.amplitude_r,
                "transverse_floquet_exponent": cycle.transverse_exponent,
                "transverse_multiplier": cycle.transverse_multiplier,
                "n_detected_maxima": cycle.n_maxima,
            }
        )
        state, previous_period = cycle.peak_state, cycle.period
        print(
            f"cycle A={A:.6f}: T={cycle.period:.9f}, "
            f"CV={cycle.period_cv:.2e}"
        )
    return rows


def aicc(residual, number_parameters):
    residual = np.asarray(residual)
    n = residual.size
    sse = float(residual @ residual)
    return (
        n * np.log(sse / n)
        + 2.0 * number_parameters
        + 2.0
        * number_parameters
        * (number_parameters + 1)
        / (n - number_parameters - 1)
    )


def fit_windows(period_rows, A_connection, lambda_u):
    A_all = np.array([row["A"] for row in period_rows])
    T_all = np.array([row["period"] for row in period_rows])

    def log_free(A, C, c, Ac):
        return C - c * np.log(Ac - A)

    def snic_free(A, C, k, Ac):
        return C + k / np.sqrt(Ac - A)

    rows = []
    for n in range(6, len(A_all) + 1):
        A, T = A_all[-n:], T_all[-n:]
        X_log = np.column_stack([np.ones(n), -np.log(A_connection - A)])
        p_log_fixed = np.linalg.lstsq(X_log, T, rcond=None)[0]
        residual_log_fixed = T - X_log @ p_log_fixed
        X_snic = np.column_stack(
            [np.ones(n), 1.0 / np.sqrt(A_connection - A)]
        )
        p_snic_fixed = np.linalg.lstsq(X_snic, T, rcond=None)[0]
        residual_snic_fixed = T - X_snic @ p_snic_fixed

        lower = float(A.max() + 1.0e-10)
        p_log_free, _ = curve_fit(
            log_free,
            A,
            T,
            p0=(p_log_fixed[0], p_log_fixed[1], A_connection),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.36]),
            maxfev=200000,
        )
        p_snic_free, _ = curve_fit(
            snic_free,
            A,
            T,
            p0=(p_snic_fixed[0], max(p_snic_fixed[1], 1.0e-8), A_connection),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.36]),
            maxfev=200000,
        )
        residual_log_free = T - log_free(A, *p_log_free)
        residual_snic_free = T - snic_free(A, *p_snic_free)
        inv_c_fixed = float(1.0 / p_log_fixed[1])
        rows.append(
            {
                "n_last_points": n,
                "A_min": float(A.min()),
                "A_max": float(A.max()),
                "A_c_manifold": A_connection,
                "inv_c_log_fixed_Ac": inv_c_fixed,
                "lambda_u_at_connection": lambda_u,
                "relative_rate_mismatch": abs(inv_c_fixed - lambda_u) / lambda_u,
                "rms_log_fixed_Ac": float(np.sqrt(np.mean(residual_log_fixed**2))),
                "rms_snic_fixed_Ac": float(
                    np.sqrt(np.mean(residual_snic_fixed**2))
                ),
                "delta_AICc_log_over_snic_fixed_Ac": float(
                    aicc(residual_snic_fixed, 2) - aicc(residual_log_fixed, 2)
                ),
                "A_c_log_free": float(p_log_free[2]),
                "inv_c_log_free": float(1.0 / p_log_free[1]),
                "A_c_snic_free": float(p_snic_free[2]),
                "rms_log_free_Ac": float(np.sqrt(np.mean(residual_log_free**2))),
                "rms_snic_free_Ac": float(np.sqrt(np.mean(residual_snic_free**2))),
                "delta_AICc_log_over_snic_free_Ac": float(
                    aicc(residual_snic_free, 3) - aicc(residual_log_free, 3)
                ),
            }
        )
    return rows


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_numeric_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def orbit_rows(connection, r_section=0.5, epsilon=1.0e-7):
    shot = manifold_shoot(
        connection, r_section, epsilon, dense_output=True
    )
    unstable = shot["unstable_solution"]
    stable_back = shot["stable_backward_solution"]
    tu = shot["unstable_flight_time"]
    ts = shot["stable_backward_flight_time"]
    time_u = np.linspace(0.0, tu, 1801)
    time_sb = np.linspace(ts, 0.0, 1201)
    xu = unstable.sol(time_u)
    xs = stable_back.sol(time_sb)
    rows = []
    for time, r, psi in zip(time_u - tu, xu[0], xu[1]):
        rows.append(
            {
                "segment": "unstable_forward_to_section",
                "time_from_section": time,
                "r": r,
                "psi": psi,
            }
        )
    for time, r, psi in zip(ts - time_sb, xs[0], xs[1]):
        rows.append(
            {
                "segment": "section_forward_to_stable",
                "time_from_section": time,
                "r": r,
                "psi": psi,
            }
        )
    return rows, shot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-periods", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    robustness = []
    for r_section in (0.4, 0.5, 0.6, 0.7):
        for epsilon in (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8):
            connection = locate_connection(r_section, epsilon)
            shot = manifold_shoot(connection, r_section, epsilon)
            robustness.append(
                {
                    "beta": BETA,
                    "r_section": r_section,
                    "epsilon": epsilon,
                    "A_connection": connection,
                    "splitting_delta_psi": shot["splitting_delta_psi"],
                    "unstable_flight_time": shot["unstable_flight_time"],
                    "stable_backward_flight_time": shot[
                        "stable_backward_flight_time"
                    ],
                }
            )
    write_rows(
        args.output_dir / "same_cut_homoclinic_manifold_robustness.csv",
        robustness,
    )
    solver_rows = solver_refinement_rows(0.5, 1.0e-7)
    write_rows(
        args.output_dir / "homoclinic_solver_convergence.csv",
        solver_rows,
    )
    # The third row is the declared production setting
    # (rtol=3e-11, atol=3e-13, max_step=0.15).
    A_connection = float(solver_rows[2]["A_connection"])
    orbit, reference_shot = orbit_rows(A_connection)
    write_rows(args.output_dir / "homoclinic_orbit_beta_0p10.csv", orbit)

    slope_step = 1.0e-8
    splitting_minus = manifold_shoot(
        A_connection - slope_step, 0.5, 1.0e-7
    )["splitting_delta_psi"]
    splitting_plus = manifold_shoot(
        A_connection + slope_step, 0.5, 1.0e-7
    )["splitting_delta_psi"]
    bracket_splitting = [
        manifold_shoot(A, 0.5, 1.0e-7)["splitting_delta_psi"]
        for A in DEFAULT_BRACKET
    ]

    saddle = reference_shot["saddle"]
    eigenvalues = np.sort(reference_shot["eigenvalues"])
    lambda_s, lambda_u = float(eigenvalues[0]), float(eigenvalues[1])
    solver_roots = np.asarray(
        [row["A_connection"] for row in solver_rows], dtype=float
    )
    tightest_root = float(solver_roots[-1])
    max_solver_deviation = float(np.max(np.abs(solver_roots - tightest_root)))
    solver_span = float(np.ptp(solver_roots))
    section_launch_span = float(
        max(row["A_connection"] for row in robustness)
        - min(row["A_connection"] for row in robustness)
    )
    # This conservative reporting allowance exceeds both observed sweeps by
    # more than an order of magnitude.  It is empirical, not a rigorous bound.
    conservative_uncertainty = 2.0e-10
    summary = {
        "beta": BETA,
        "A_H": 0.2777754846476286,
        "raw_root": A_connection,
        "publication_value": round(A_connection, 6),
        "A_homoclinic_manifold": A_connection,
        "reported_precision_recommendation": 0.353697,
        "solver_refinement_reference_root": tightest_root,
        "solver_refinement_root_range": [
            float(np.min(solver_roots)),
            float(np.max(solver_roots)),
        ],
        "solver_refinement_span": solver_span,
        "maximum_solver_deviation_from_tightest": max_solver_deviation,
        "section_launch_root_span": section_launch_span,
        "conservative_empirical_parameter_uncertainty": (
            conservative_uncertainty
        ),
        "uncertainty_status": (
            "Empirical sensitivity allowance from solver, section, and launch "
            "refinement; not a rigorous or computer-assisted error bound."
        ),
        "saddle": saddle.tolist(),
        "lambda_u": lambda_u,
        "lambda_s": lambda_s,
        "inverse_lambda_u": 1.0 / lambda_u,
        "saddle_quantity": lambda_u + lambda_s,
        "reference_section_r": 0.5,
        "reference_epsilon": 1.0e-7,
        "reference_section_splitting": reference_shot["splitting_delta_psi"],
        "default_bracket_A": list(DEFAULT_BRACKET),
        "default_bracket_splitting_delta_psi": bracket_splitting,
        "centered_splitting_slope_step": slope_step,
        "centered_d_splitting_dA": (
            splitting_plus - splitting_minus
        ) / (2.0 * slope_step),
        "range_Ac_over_sections_and_epsilons": [
            float(min(row["A_connection"] for row in robustness)),
            float(max(row["A_connection"] for row in robustness)),
        ],
        "method": (
            "Forward shooting of the lower-r unstable saddle branch to its "
            "return crossing, matched to backward shooting of the lower-r "
            "stable branch on r=0.5; robustness checked on four sections and "
            "four launch distances. DOP853 tolerances and maximum step were "
            "also refined independently; full precision is retained for "
            "reproduction while the publication value is rounded to six "
            "decimal places."
        ),
    }

    if not args.skip_periods:
        periods = period_series()
        write_rows(
            args.output_dir / "homoclinic_periods_beta_0p10.csv", periods
        )
    else:
        period_path = args.output_dir / "homoclinic_periods_beta_0p10.csv"
        if not period_path.is_file():
            raise RuntimeError(
                "--skip-periods requires the archived table "
                f"{period_path}"
            )
        periods = read_numeric_rows(period_path)

    fits = fit_windows(periods, A_connection, lambda_u)
    write_rows(
        args.output_dir / "homoclinic_fit_windows_beta_0p10.csv", fits
    )
    summary["six_point_asymptotic_fit"] = fits[0]
    summary["all_point_fit"] = fits[-1]

    (args.output_dir / "same_cut_homoclinic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
