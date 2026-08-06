#!/usr/bin/env python3
"""Local bifurcation, relaxation, Floquet, and homoclinic calculations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import expm
from scipy.optimize import curve_fit, root
from scipy.signal import find_peaks

from model import (
    codimension_one_point,
    two_by_two_decomposition,
    two_pop_equilibrium,
    two_pop_jacobian,
    two_pop_rhs,
    two_pop_transverse_rate,
)


Array = np.ndarray


def declared_two_pop_block(x: Array, J: Array) -> Array:
    """Represent J in the declared order-parameter metric.

    On ``rho_1=1`` and after quotienting global phase, the metric in
    ``(r,psi)`` coordinates is ``diag(1,r^2/(1+r^2))``.
    """
    r = float(x[0])
    C = np.diag([1.0, r / np.sqrt(1.0 + r * r)])
    return C @ J @ np.linalg.inv(C)


def benchmark_points(beta: float = 0.10) -> dict[str, dict[str, object]]:
    """High-accuracy SN, EP, Hopf, and Bogdanov--Takens benchmarks."""
    points = {
        "SN": codimension_one_point(beta, "SN", (0.821, -0.218, 0.17834)),
        "EP": codimension_one_point(beta, "EP", (0.815, -0.217, 0.178454)),
        "Hopf": codimension_one_point(beta, "Hopf", (0.589, -0.204, 0.277775)),
    }

    def bt_equations(y: Array) -> Array:
        x, A, b = y[:2], float(y[2]), float(y[3])
        J = two_pop_jacobian(x, A, b)
        return np.r_[
            two_pop_rhs(0.0, x, A, b), np.linalg.det(J), np.trace(J)
        ]

    bt = root(bt_equations, np.array([0.659, -np.pi / 6.0, 0.3372, 0.2239]), tol=1e-12)
    if np.linalg.norm(bt_equations(bt.x), ord=np.inf) > 1.0e-9:
        raise RuntimeError("Bogdanov--Takens solve failed")
    points["BT"] = {
        "kind": "BT",
        "r": float(bt.x[0]),
        "psi": float(bt.x[1]),
        "A": float(bt.x[2]),
        "beta": float(bt.x[3]),
        "J": two_pop_jacobian(bt.x[:2], bt.x[2], bt.x[3]),
    }
    return points


def stable_locked_branch(beta: float, A_values: Array) -> list[dict[str, object]]:
    """Continue the physical locked branch from large to small A.

    Descending continuation prevents Newton from jumping to the nearby saddle
    branch in the narrow interval between the saddle-node and EP.
    """
    A_values = np.asarray(A_values, float)
    order = np.argsort(A_values)[::-1]
    guess = np.array([0.50, -2.0 * beta])
    records: dict[int, dict[str, object]] = {}
    for idx in order:
        A = float(A_values[idx])
        x = two_pop_equilibrium(A, beta, guess)
        guess = x
        J = two_pop_jacobian(x, A, beta)
        eig = np.linalg.eigvals(J)
        B = declared_two_pop_block(x, J)
        diag = two_by_two_decomposition(B)
        records[idx] = {
            "A": A,
            "x": x.copy(),
            "J": J,
            "B_declared_metric": B,
            "eigenvalues": eig,
            "lambda_transverse": two_pop_transverse_rate(x, A, beta),
            **diag,
        }
    return [records[i] for i in range(len(A_values))]


def adaptive_branch(beta: float = 0.10) -> tuple[list[dict[str, object]], dict]:
    """Resolve the very short SN--EP interval and the full route to Hopf."""
    events = benchmark_points(beta)
    A_sn = float(events["SN"]["A"])
    A_ep = float(events["EP"]["A"])
    A_h = float(events["Hopf"]["A"])
    near = np.linspace(A_sn + 2.0e-8, A_ep + 8.0e-4, 220)
    broad = np.linspace(A_ep + 8.0e-4, A_h + 0.035, 300)
    A_values = np.unique(np.r_[near, broad, A_sn, A_ep, A_h])
    # The exact SN point is solved independently; continuation starts above it.
    usable = A_values[A_values > A_sn + 1.0e-10]
    return stable_locked_branch(beta, usable), events


def relaxation_pulse_data(
    beta: float = 0.10,
    kick: float = 1.0e-4,
    t_end: float = 260.0,
) -> dict[str, object]:
    """Pulse-response spectroscopy below, at, and above the EP.

    A small coherence kick is applied.  The nonlinear OA response is
    compared with ``exp(Jt)``.  The observable is the coherence displacement
    ``delta r`` divided by the kick amplitude.
    """
    events = benchmark_points(beta)
    ep = events["EP"]
    A_ep = float(ep["A"])
    A_sn = float(events["SN"]["A"])
    sn_ep_gap = A_ep - A_sn
    cases = [
        ("stable node", A_ep - 0.45 * sn_ep_gap),
        ("critical damping", A_ep),
        ("stable focus", A_ep + 12.0 * sn_ep_gap),
    ]
    time = np.linspace(0.0, t_end, 1401)
    out: dict[str, object] = {"time": time, "cases": []}
    guess = np.array([float(ep["r"]), float(ep["psi"])])
    for label, A in cases:
        xeq = two_pop_equilibrium(A, beta, guess)
        guess = xeq
        J = two_pop_jacobian(xeq, A, beta)
        # A coherence pulse gives directly readable closed forms for delta r.
        perturb = np.array([kick, 0.0])
        linear = np.array([(expm(J * t) @ perturb)[0] / kick for t in time])
        sol = solve_ivp(
            lambda t, x: two_pop_rhs(t, x, A, beta),
            (time[0], time[-1]),
            xeq + perturb,
            t_eval=time,
            method="DOP853",
            rtol=2.0e-11,
            atol=2.0e-13,
            max_step=0.25,
        )
        nonlinear = (sol.y[0] - xeq[0]) / kick
        out["cases"].append(
            {
                "label": label,
                "A": float(A),
                "x": xeq,
                "J": J,
                "eigenvalues": np.linalg.eigvals(J),
                "linear_dr_over_kick": linear,
                "nonlinear_dr_over_kick": nonlinear,
            }
        )
    return out


def derivative_tensors_poly(
    x: Array,
    A: float,
    beta: float,
    h: float = 0.02,
    degree: int = 7,
    ngrid: int = 9,
) -> tuple[Array, Array]:
    """Second/third derivatives from a local high-order polynomial fit."""
    exponents = [
        (i, j) for i in range(degree + 1) for j in range(degree + 1 - i)
    ]
    grid = np.linspace(-1.0, 1.0, ngrid)
    points = np.asarray([(u, v) for u in grid for v in grid])
    design = np.asarray(
        [[u**i * v**j for i, j in exponents] for u, v in points]
    )
    values = np.asarray(
        [two_pop_rhs(0.0, x + h * np.asarray([u, v]), A, beta) for u, v in points]
    )
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    loc = {exp: k for k, exp in enumerate(exponents)}
    H = np.zeros((2, 2, 2))
    T = np.zeros((2, 2, 2, 2))
    for output in range(2):
        H[output, 0, 0] = 2.0 * coefficients[loc[(2, 0)], output] / h**2
        H[output, 0, 1] = H[output, 1, 0] = (
            coefficients[loc[(1, 1)], output] / h**2
        )
        H[output, 1, 1] = 2.0 * coefficients[loc[(0, 2)], output] / h**2
        for indices in np.ndindex(2, 2, 2):
            i = indices.count(0)
            j = 3 - i
            T[(output,) + indices] = (
                coefficients[loc[(i, j)], output]
                * math.factorial(i)
                * math.factorial(j)
                / h**3
            )
    return H, T


def first_lyapunov(beta: float) -> dict[str, object]:
    """First Lyapunov coefficient and its three standard contributions."""
    hopf = codimension_one_point(beta, "Hopf", (0.59, -2.05 * beta, 0.28))
    A_h = float(hopf["A"])
    x_h = np.array([hopf["r"], hopf["psi"]])
    J = two_pop_jacobian(x_h, A_h, beta)
    eig, right = np.linalg.eig(J)
    iq = int(np.argmax(eig.imag))
    omega = float(eig[iq].imag)
    q = right[:, iq]
    eig_left, left = np.linalg.eig(J.T)
    ip = int(np.argmin(abs(eig_left + 1j * omega)))
    p = left[:, ip]
    p = p / np.conj(np.vdot(p, q))
    H, T = derivative_tensors_poly(x_h, A_h, beta)
    bilinear = lambda u, v: np.einsum("ijk,j,k->i", H, u, v)
    trilinear = lambda u, v, w: np.einsum("ijkl,j,k,l->i", T, u, v, w)
    direct = np.vdot(p, trilinear(q, q, np.conj(q)))
    zero = -2.0 * np.vdot(
        p, bilinear(q, np.linalg.solve(J, bilinear(q, np.conj(q))))
    )
    second = np.vdot(
        p,
        bilinear(
            np.conj(q),
            np.linalg.solve(2j * omega * np.eye(2) - J, bilinear(q, q)),
        ),
    )
    terms = np.real(np.asarray([direct, zero, second])) / (2.0 * omega)
    return {
        "beta": float(beta),
        "A_H": A_h,
        "Omega_H": omega,
        "l1": float(np.sum(terms)),
        "direct_cubic": float(terms[0]),
        "zero_frequency_feedback": float(terms[1]),
        "second_harmonic": float(terms[2]),
    }


def transverse_cycle(
    A: float,
    beta: float,
    transient: float | None = None,
) -> dict[str, object]:
    """Period and transverse Floquet exponent of an attracting breather."""
    if transient is None:
        transient = 6500.0 if A < 0.39 else 12000.0
    # The equilibrium is unstable above Hopf; this merely supplies a nearby start.
    guess = np.array([0.59, -2.05 * beta])
    xeq = two_pop_equilibrium(A, beta, guess)
    x0 = xeq + np.array([0.01, 0.01])

    def maximum_event(t: float, x: Array) -> float:
        return float(two_pop_rhs(t, x, A, beta)[0])

    maximum_event.direction = -1
    maximum_event.terminal = False
    sol = solve_ivp(
        lambda t, x: two_pop_rhs(t, x, A, beta),
        (0.0, transient),
        x0,
        method="DOP853",
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step=0.20,
        dense_output=True,
        events=maximum_event,
    )
    peaks = sol.t_events[0]
    peaks = peaks[peaks > transient * 0.55]
    if peaks.size < 3:
        raise RuntimeError(f"too few cycle maxima at A={A:g}, beta={beta:g}")
    t0, t1 = float(peaks[-2]), float(peaks[-1])
    period = t1 - t0
    time = np.linspace(t0, t1, 4001)
    orbit = sol.sol(time)
    rates = np.asarray(
        [two_pop_transverse_rate(orbit[:, k], A, beta) for k in range(time.size)]
    )
    traces = np.asarray(
        [np.trace(two_pop_jacobian(orbit[:, k], A, beta)) for k in range(time.size)]
    )
    exponent = float(trapezoid(rates, time) / period)
    inplane_exponent = float(trapezoid(traces, time) / period)
    return {
        "A": float(A),
        "beta": float(beta),
        "period": float(period),
        "lambda_transverse_Floquet": exponent,
        "multiplier_transverse": float(np.exp(exponent * period)),
        "lambda_inplane_Floquet": inplane_exponent,
        "multiplier_inplane": float(np.exp(inplane_exponent * period)),
        "multiplier_trivial": 1.0,
        "amplitude_r": float((np.max(orbit[0]) - np.min(orbit[0])) / 2.0),
        "orbit_r": orbit[0],
        "orbit_psi": orbit[1],
        "orbit_time": time - t0,
    }


def measure_period_series(
    beta: float = 0.06,
    A_values: Array | None = None,
) -> list[dict[str, object]]:
    """Recompute the period series used in the global-bifurcation fit."""
    if A_values is None:
        A_values = np.array(
            [0.320, 0.325, 0.335, 0.350, 0.365, 0.375,
             0.385, 0.390, 0.392, 0.394, 0.395, 0.396]
        )
    return [transverse_cycle(float(A), beta) for A in np.asarray(A_values)]


def fit_homoclinic_windows(period_data: Array, beta: float = 0.06) -> list[dict]:
    """Compare logarithmic and SNIC fits for every last-n-point window."""
    period_data = np.asarray(period_data, float)
    Aall, Tall = period_data[:, 0], period_data[:, 1]

    def log_model(A: Array, C: float, c: float, Ac: float) -> Array:
        return C - c * np.log(Ac - A)

    def snic_model(A: Array, C: float, k: float, Ac: float) -> Array:
        return C + k / np.sqrt(Ac - A)

    def aicc(residual: Array, k: int = 3) -> float:
        n = residual.size
        sse = np.sum(residual * residual)
        return float(
            n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        )

    rows = []
    # n>=6 is mathematically permitted for k=3; n>=7 is the conservative plot.
    for n in range(6, len(Aall) + 1):
        A, T = Aall[-n:], Tall[-n:]
        lower = float(A.max() + 1.0e-8)
        plog, covlog = curve_fit(
            log_model,
            A,
            T,
            p0=(-40.0, 31.0, 0.39616),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.45]),
            maxfev=200000,
        )
        psnic, _ = curve_fit(
            snic_model,
            A,
            T,
            p0=(20.0, 7.0, 0.397),
            bounds=([-5000.0, 0.0, lower], [5000.0, 5000.0, 0.45]),
            maxfev=200000,
        )
        residual_log = T - log_model(A, *plog)
        residual_snic = T - snic_model(A, *psnic)
        Ac = float(plog[2])
        saddle = two_pop_equilibrium(Ac, beta, (0.982, -0.197))
        eig = np.linalg.eigvals(two_pop_jacobian(saddle, Ac, beta))
        lam_u, lam_s = float(np.max(eig.real)), float(np.min(eig.real))
        inv_c = float(1.0 / plog[1])
        rows.append(
            {
                "n_last_points": n,
                "A_min": float(A.min()),
                "A_c_log": Ac,
                "inv_c_log": inv_c,
                "inv_c_standard_error": float(
                    np.sqrt(np.diag(covlog))[1] / plog[1] ** 2
                ),
                "lambda_u": lam_u,
                "lambda_s": lam_s,
                "saddle_quantity": lam_u + lam_s,
                "relative_rate_mismatch": abs(inv_c - lam_u) / lam_u,
                "delta_AICc_log_over_snic": aicc(residual_snic) - aicc(residual_log),
                "rms_log": float(np.sqrt(np.mean(residual_log**2))),
                "rms_snic": float(np.sqrt(np.mean(residual_snic**2))),
                "log_parameters": plog,
                "snic_parameters": psnic,
            }
        )
    return rows


def trace_codimension_loci(beta_values: Array) -> dict[str, list[dict[str, object]]]:
    """Continue SN, EP, and Hopf loci from beta=0.10 in both directions."""
    beta_values = np.asarray(beta_values, float)
    center = int(np.argmin(abs(beta_values - 0.10)))
    seeds = {
        "SN": np.array([0.821, -0.218, 0.17834]),
        "EP": np.array([0.815, -0.217, 0.178454]),
        "Hopf": np.array([0.589, -0.204, 0.277775]),
    }
    out: dict[str, list[dict[str, object]]] = {}
    for kind, seed in seeds.items():
        records: dict[int, dict[str, object]] = {}
        p0 = codimension_one_point(float(beta_values[center]), kind, tuple(seed))
        records[center] = p0
        last = np.array([p0["r"], p0["psi"], p0["A"]], float)
        for i in range(center - 1, -1, -1):
            p = codimension_one_point(float(beta_values[i]), kind, tuple(last))
            records[i] = p
            last = np.array([p["r"], p["psi"], p["A"]], float)
        last = np.array([p0["r"], p0["psi"], p0["A"]], float)
        for i in range(center + 1, beta_values.size):
            p = codimension_one_point(float(beta_values[i]), kind, tuple(last))
            records[i] = p
            last = np.array([p["r"], p["psi"], p["A"]], float)
        out[kind] = [records[i] for i in range(beta_values.size)]
    return out
