#!/usr/bin/env python3
"""Robustness of the node--focus/EP ordering to frequency dispersion.

The main benchmark has identical oscillators (Delta=0), for which the
Watanabe--Strogatz constants prevent generic attraction to the
Ott--Antonsen (OA) manifold.  This script continues the saddle-node,
defective node--focus transition, and Hopf bifurcation in the complete
three-dimensional two-population OA phase quotient for small positive
Lorentzian half-width Delta.

It also:

* compares linear and nonlinear coherence-pulse responses at Delta=1e-3; and
* verifies an oscillator-level Möbius phase-reset realization of the
  amplitude pulse used in the reduced equations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import root

from model import (
    complex_step_jacobian,
    couplings,
    oa_quotient_rhs,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BETA = 0.10


def full_rhs(x: np.ndarray, A: float, Delta: float) -> np.ndarray:
    """Two-population OA quotient in x=(rho1,rho2,phi1-phi2)."""
    mu, nu = couplings(A)
    K = np.array([[mu, nu], [nu, mu]], dtype=float)
    return oa_quotient_rhs(0.0, x, K, BETA, Delta)


def full_jacobian(x: np.ndarray, A: float, Delta: float) -> np.ndarray:
    return complex_step_jacobian(lambda y: full_rhs(y, A, Delta), x)


def equilibrium(
    A: float,
    Delta: float,
    guess: np.ndarray,
) -> np.ndarray:
    sol = root(lambda x: full_rhs(x, A, Delta), np.asarray(guess, float), tol=1e-12)
    residual = float(np.linalg.norm(full_rhs(sol.x, A, Delta), ord=np.inf))
    if residual > 1e-9:
        raise RuntimeError(
            f"equilibrium solve failed at A={A:.8g}, Delta={Delta:.8g}; "
            f"residual={residual:.3e}"
        )
    return np.asarray(sol.x, float)


def characteristic_coefficients(J: np.ndarray) -> np.ndarray:
    """Return [1,c1,c2,c3] for det(lambda I-J)."""
    return np.real_if_close(np.poly(J)).astype(float)


def cubic_discriminant(coefficients: np.ndarray) -> float:
    """Discriminant of x^3+b*x^2+c*x+d."""
    _, b, c, d = coefficients
    return float(
        b * b * c * c
        - 4.0 * c**3
        - 4.0 * b**3 * d
        - 27.0 * d * d
        + 18.0 * b * c * d
    )


def solve_saddle_node(Delta: float, guess: np.ndarray) -> dict[str, object]:
    def equations(y: np.ndarray) -> np.ndarray:
        x, A = y[:3], float(y[3])
        J = full_jacobian(x, A, Delta)
        return np.r_[full_rhs(x, A, Delta), np.linalg.det(J)]

    sol = root(equations, np.asarray(guess, float), tol=1e-12)
    residual = float(np.linalg.norm(equations(sol.x), ord=np.inf))
    if residual > 1e-9:
        raise RuntimeError(
            f"saddle-node solve failed at Delta={Delta:.8g}; residual={residual:.3e}"
        )
    x, A = np.asarray(sol.x[:3], float), float(sol.x[3])
    J = full_jacobian(x, A, Delta)
    return {
        "x": x,
        "A": A,
        "J": J,
        "eigenvalues": np.linalg.eigvals(J),
        "residual_inf": residual,
        "guess": np.r_[x, A],
    }


def solve_exceptional_point(Delta: float, guess: np.ndarray) -> dict[str, object]:
    """Solve p(lambda)=p'(lambda)=0 together with the locked-state equations."""

    def equations(y: np.ndarray) -> np.ndarray:
        x, A, lam = y[:3], float(y[3]), float(y[4])
        coefficients = characteristic_coefficients(full_jacobian(x, A, Delta))
        _, c1, c2, c3 = coefficients
        p = ((lam + c1) * lam + c2) * lam + c3
        dp = 3.0 * lam * lam + 2.0 * c1 * lam + c2
        return np.r_[full_rhs(x, A, Delta), p, dp]

    sol = root(equations, np.asarray(guess, float), tol=1e-12)
    residual = float(np.linalg.norm(equations(sol.x), ord=np.inf))
    if residual > 1e-8:
        raise RuntimeError(
            f"EP solve failed at Delta={Delta:.8g}; residual={residual:.3e}"
        )
    x, A, lam = (
        np.asarray(sol.x[:3], float),
        float(sol.x[3]),
        float(sol.x[4]),
    )
    J = full_jacobian(x, A, Delta)
    singular_values = np.linalg.svd(J - lam * np.eye(3), compute_uv=False)
    eigenvalues = np.linalg.eigvals(J)
    third = float(np.trace(J) - 2.0 * lam)
    return {
        "x": x,
        "A": A,
        "lambda_repeated": lam,
        "lambda_third": third,
        "J": J,
        "eigenvalues": eigenvalues,
        "singular_values_shifted": singular_values,
        "spectral_abscissa": float(np.max(eigenvalues.real)),
        "residual_inf": residual,
        "guess": np.r_[x, A, lam],
    }


def solve_hopf(Delta: float, guess: np.ndarray) -> dict[str, object]:
    """Use c3=c1*c2 for one real root and one imaginary conjugate pair."""

    def equations(y: np.ndarray) -> np.ndarray:
        x, A = y[:3], float(y[3])
        _, c1, c2, c3 = characteristic_coefficients(
            full_jacobian(x, A, Delta)
        )
        return np.r_[full_rhs(x, A, Delta), c3 - c1 * c2]

    sol = root(equations, np.asarray(guess, float), tol=1e-12)
    residual = float(np.linalg.norm(equations(sol.x), ord=np.inf))
    if residual > 1e-9:
        raise RuntimeError(
            f"Hopf solve failed at Delta={Delta:.8g}; residual={residual:.3e}"
        )
    x, A = np.asarray(sol.x[:3], float), float(sol.x[3])
    J = full_jacobian(x, A, Delta)
    eigenvalues = np.linalg.eigvals(J)
    pair = eigenvalues[np.argsort(np.abs(eigenvalues.real))[:2]]
    return {
        "x": x,
        "A": A,
        "J": J,
        "eigenvalues": eigenvalues,
        "Omega": float(np.mean(np.abs(pair.imag))),
        "residual_inf": residual,
        "guess": np.r_[x, A],
    }


def continue_events(deltas: np.ndarray) -> tuple[list[dict[str, float]], dict]:
    sn_guess = np.array([1.0, 0.82064244, -0.21776426, 0.17834090])
    ep_guess = np.array(
        [1.0, 0.81503449, -0.21698330, 0.17845421, -0.02276564]
    )
    hopf_guess = np.array([1.0, 0.58948031, -0.20430746, 0.27777548])
    rows: list[dict[str, float]] = []
    details: dict[str, dict[str, object]] = {}

    for Delta in np.asarray(deltas, float):
        sn = solve_saddle_node(float(Delta), sn_guess)
        ep = solve_exceptional_point(float(Delta), ep_guess)
        hopf = solve_hopf(float(Delta), hopf_guess)
        sn_guess, ep_guess, hopf_guess = sn["guess"], ep["guess"], hopf["guess"]
        if not (sn["A"] < ep["A"] < hopf["A"]):
            raise RuntimeError(f"event ordering failed at Delta={Delta:g}")
        sv = np.asarray(ep["singular_values_shifted"])
        rows.append(
            {
                "Delta": float(Delta),
                "A_SN": float(sn["A"]),
                "A_EP": float(ep["A"]),
                "A_Hopf": float(hopf["A"]),
                "SN_to_EP": float(ep["A"] - sn["A"]),
                "EP_to_Hopf": float(hopf["A"] - ep["A"]),
                "lambda_EP": float(ep["lambda_repeated"]),
                "lambda_other_EP": float(ep["lambda_third"]),
                "spectral_abscissa_EP": float(ep["spectral_abscissa"]),
                "sv1_shifted": float(sv[0]),
                "sv2_shifted": float(sv[1]),
                "sv3_shifted": float(sv[2]),
                "Omega_Hopf": float(hopf["Omega"]),
                "residual_SN": float(sn["residual_inf"]),
                "residual_EP": float(ep["residual_inf"]),
                "residual_Hopf": float(hopf["residual_inf"]),
            }
        )
        details[f"{Delta:.8g}"] = {"SN": sn, "EP": ep, "Hopf": hopf}
    return rows, details


def pulse_responses(
    event_details: dict[str, object],
    Delta: float,
    epsilon: float = 1e-4,
) -> list[dict[str, float]]:
    sn = event_details["SN"]
    ep = event_details["EP"]
    gap = float(ep["A"] - sn["A"])
    cases = [
        ("node", float(ep["A"] - 0.45 * gap)),
        ("EP", float(ep["A"])),
        ("focus", float(ep["A"] + 12.0 * gap)),
    ]
    time = np.linspace(0.0, 300.0, 1501)
    records: list[dict[str, float]] = []
    guess = np.asarray(ep["x"], float)
    for label, A in cases:
        xeq = equilibrium(A, Delta, guess)
        guess = xeq
        J = full_jacobian(xeq, A, Delta)
        perturb = np.array([0.0, epsilon, 0.0])
        linear = np.array([(expm(J * t) @ perturb)[1] / epsilon for t in time])
        sol = solve_ivp(
            lambda t, x: full_rhs(x, A, Delta),
            (float(time[0]), float(time[-1])),
            xeq + perturb,
            t_eval=time,
            method="DOP853",
            rtol=2e-11,
            atol=2e-13,
            max_step=0.25,
        )
        nonlinear = (sol.y[1] - xeq[1]) / epsilon
        for t, lin, nonlin in zip(time, linear, nonlinear):
            records.append(
                {
                    "Delta": Delta,
                    "case": label,
                    "A": A,
                    "time": float(t),
                    "linear_drho2_over_epsilon": float(lin),
                    "nonlinear_drho2_over_epsilon": float(nonlin),
                }
            )
    return records


def mobius_reset(phases: np.ndarray, target_amplitude: float) -> np.ndarray:
    """Reset phases so an OA Poisson state changes amplitude at fixed phase.

    For u=exp(i(theta-phi)), u+ = (u+a)/(1+a*u), where
    a=(r+-r)/(1-r*r+).  This disk automorphism maps the Poisson-kernel
    parameter r to r+ while preserving the mean phase.
    """
    z = np.mean(np.exp(1j * phases))
    phi = float(np.angle(z))
    r = float(abs(z))
    a = float((target_amplitude - r) / (1.0 - r * target_amplitude))
    if abs(a) >= 1.0:
        raise ValueError("requested reset is outside the disk automorphism range")
    u = np.exp(1j * (phases - phi))
    reset = (u + a) / (1.0 + a * u)
    return phi + np.angle(reset)


def poisson_sample(r: float, size: int, rng: np.random.Generator) -> np.ndarray:
    uniform = np.exp(1j * rng.uniform(-np.pi, np.pi, size))
    sample = (uniform + r) / (1.0 + r * uniform)
    return np.angle(sample)


def microscopic_pulse_validation(
    r_reference: float,
    epsilon: float = 1e-3,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(20260727)
    rows: list[dict[str, float]] = []
    for N in (128, 256, 512, 1024, 2048, 4096):
        for seed_index in range(12):
            phases = poisson_sample(r_reference, N, rng)
            z_before = np.mean(np.exp(1j * phases))
            target = float(abs(z_before) + epsilon)
            reset = mobius_reset(phases, target)
            z_after = np.mean(np.exp(1j * reset))
            rows.append(
                {
                    "N": N,
                    "seed_index": seed_index,
                    "r_before": float(abs(z_before)),
                    "target_r_after": target,
                    "r_after": float(abs(z_after)),
                    "amplitude_error": float(abs(z_after) - target),
                    "phase_shift": float(
                        np.angle(np.exp(1j * (np.angle(z_after) - np.angle(z_before))))
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items() if key != "J"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    DATA.mkdir(exist_ok=True)
    deltas = np.linspace(0.0, 0.005, 21)
    event_rows, details = continue_events(deltas)
    write_csv(DATA / "dispersive_event_loci.csv", event_rows)

    pulse_delta = 0.001
    pulse_key = f"{pulse_delta:.8g}"
    pulses = pulse_responses(details[pulse_key], pulse_delta)
    write_csv(DATA / "dispersive_pulse_responses.csv", pulses)

    ep_state = np.asarray(details[pulse_key]["EP"]["x"], float)
    microscopic = microscopic_pulse_validation(float(ep_state[1]))
    write_csv(DATA / "microscopic_pulse_validation.csv", microscopic)

    max_amp_error = max(abs(row["amplitude_error"]) for row in microscopic)
    max_phase_shift = max(abs(row["phase_shift"]) for row in microscopic)
    summary = {
        "beta": BETA,
        "Delta_range": [float(deltas.min()), float(deltas.max())],
        "number_of_Delta_values": int(deltas.size),
        "ordering_verified": "A_SN < A_EP < A_Hopf at every sampled Delta",
        "Delta_0": event_rows[0],
        "Delta_0p001": event_rows[4],
        "Delta_0p005": event_rows[-1],
        "pulse_Delta": pulse_delta,
        "pulse_epsilon": 1e-4,
        "microscopic_reset": {
            "map": "u_plus=(u+a)/(1+a*u), a=(r_plus-r)/(1-r*r_plus)",
            "epsilon": 1e-3,
            "maximum_finite_N_amplitude_error": max_amp_error,
            "maximum_finite_N_phase_shift": max_phase_shift,
        },
        "event_details_at_Delta_0p001": json_safe(details[pulse_key]),
    }
    (DATA / "dispersive_robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
