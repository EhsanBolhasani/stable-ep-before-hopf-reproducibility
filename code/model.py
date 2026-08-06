#!/usr/bin/env python3
"""Core Ott--Antonsen models used by the reproducibility repository.

The two-population coordinates are ``x=(r, psi)`` with
``psi=phi_1-phi_2`` and ``rho_1=1``.  For a general M-population
network the global phase is removed and

    x=(rho_1,...,rho_M, psi_1,...,psi_{M-1}),

where ``psi_sigma=phi_sigma-phi_M``.  All phase-lag parameters called
``beta`` below satisfy ``alpha=pi/2-beta`` in the microscopic
Kuramoto--Sakaguchi equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root


Array = np.ndarray


def couplings(A: float) -> tuple[float, float]:
    """Return the conventional self/cross couplings ``mu, nu``."""
    return (1.0 + A) / 2.0, (1.0 - A) / 2.0


def two_pop_rhs(_t: float, x: Array, A: float, beta: float) -> Array:
    """Invariant-boundary two-population OA equations."""
    r, psi = x
    mu, nu = couplings(A)
    sb, cb = np.sin(beta), np.cos(beta)
    h = (1.0 + r * r) / (2.0 * r)
    dr = (1.0 - r * r) / 2.0 * (
        mu * r * sb + nu * np.sin(psi + beta)
    )
    dpsi = (
        -mu * cb
        - nu * r * np.cos(beta - psi)
        + h * (mu * r * cb + nu * np.cos(psi + beta))
    )
    return np.asarray([dr, dpsi], dtype=np.result_type(x, float))


def two_pop_jacobian(x: Array, A: float, beta: float) -> Array:
    """Analytic Jacobian of :func:`two_pop_rhs` in ``(r,psi)``."""
    r, psi = x
    mu, nu = couplings(A)
    sb, cb = np.sin(beta), np.cos(beta)
    h = (1.0 + r * r) / (2.0 * r)
    hp = (1.0 - 1.0 / (r * r)) / 2.0
    U = mu * r * sb + nu * np.sin(psi + beta)
    V = mu * r * cb + nu * np.cos(psi + beta)
    f_r = -r * U + (1.0 - r * r) * mu * sb / 2.0
    f_psi = (1.0 - r * r) * nu * np.cos(psi + beta) / 2.0
    g_r = -nu * np.cos(beta - psi) + hp * V + h * mu * cb
    g_psi = -nu * r * np.sin(beta - psi) - h * nu * np.sin(psi + beta)
    return np.asarray([[f_r, f_psi], [g_r, g_psi]], dtype=float)


def two_pop_equilibrium(
    A: float,
    beta: float,
    guess: Array | tuple[float, float] = (0.6, -0.2),
) -> Array:
    """Solve the locked-state equations with an analytic Newton matrix."""
    sol = root(
        lambda y: two_pop_rhs(0.0, y, A, beta),
        np.asarray(guess, float),
        jac=lambda y: two_pop_jacobian(y, A, beta),
        tol=1.0e-12,
    )
    residual = np.linalg.norm(two_pop_rhs(0.0, sol.x, A, beta), ord=np.inf)
    if residual > 1.0e-9:
        raise RuntimeError(
            f"locked-state solve failed at A={A:g}, beta={beta:g}; "
            f"residual={residual:.3e}"
        )
    return sol.x


def two_pop_transverse_rate(x: Array, A: float, beta: float) -> float:
    """Instantaneous exponent normal to the boundary ``rho_1=1``."""
    r, psi = x
    mu, nu = couplings(A)
    return float(-(mu * np.sin(beta) + nu * r * np.sin(beta - psi)))


def codimension_one_point(
    beta: float,
    kind: str,
    guess: tuple[float, float, float],
) -> dict[str, object]:
    """Locate a saddle-node, exceptional point, or Hopf point.

    ``kind`` is one of ``"SN"``, ``"EP"``, and ``"Hopf"``.  The
    augmented equations are respectively ``det(J)=0``, ``D=0``, and
    ``tr(J)=0``.
    """

    def equations(y: Array) -> Array:
        x, A = np.asarray(y[:2]), float(y[2])
        J = two_pop_jacobian(x, A, beta)
        if kind == "SN":
            condition = np.linalg.det(J)
        elif kind == "EP":
            condition = np.trace(J) ** 2 - 4.0 * np.linalg.det(J)
        elif kind == "Hopf":
            condition = np.trace(J)
        else:
            raise ValueError(f"unknown codimension-one point: {kind}")
        return np.r_[two_pop_rhs(0.0, x, A, beta), condition]

    sol = root(equations, np.asarray(guess, float), tol=1.0e-12)
    residual = float(np.linalg.norm(equations(sol.x), ord=np.inf))
    if residual > 1.0e-9:
        raise RuntimeError(
            f"{kind} solve failed at beta={beta:g}; residual={residual:.3e}"
        )
    x, A = np.asarray(sol.x[:2]), float(sol.x[2])
    J = two_pop_jacobian(x, A, beta)
    eig = np.linalg.eigvals(J)
    return {
        "kind": kind,
        "A": A,
        "r": float(x[0]),
        "psi": float(x[1]),
        "J": J,
        "eig": eig,
        "discriminant": float(np.trace(J) ** 2 - 4.0 * np.linalg.det(J)),
        "lambda_transverse": two_pop_transverse_rate(x, A, beta),
        "residual_inf": residual,
    }


def two_by_two_decomposition(B: Array) -> dict[str, float]:
    """Return invariant scalar diagnostics for a real 2 x 2 block.

    Write ``B=tau I + [[delta,s+g],[s-g,-delta]]``.  The bounded
    modal-skewness measure is

        R_B=|g|/sqrt(delta**2+s**2+g**2).

    It is invariant under orthogonal changes of basis within the selected
    plane and uniform time rescaling.  It is not invariant under arbitrary
    non-orthogonal rescaling of state coordinates.
    """
    B = np.asarray(B, float)
    a, b = B[0]
    c, d = B[1]
    tau = (a + d) / 2.0
    delta = (a - d) / 2.0
    s = (b + c) / 2.0
    g = (b - c) / 2.0
    gcrit = float(np.hypot(delta, s))
    scale = float(np.sqrt(delta * delta + s * s + g * g))
    R_B = abs(g) / scale if scale > 0.0 else 0.0
    D = float(4.0 * (delta * delta + s * s - g * g))
    return {
        "tau": float(tau),
        "delta": float(delta),
        "s": float(s),
        "g": float(g),
        "gcrit": gcrit,
        "R_B": float(R_B),
        "bounded_margin": float(2.0 * R_B * R_B - 1.0),
        "discriminant": D,
    }


def state_weighted_interactions(
    rho: Array,
    phi: Array,
    K: Array,
    beta: float | Array,
) -> tuple[Array, Array]:
    """Return cosine- and sine-weighted interactions ``X,Y``."""
    phase = phi[None, :] - phi[:, None] + np.asarray(beta)
    X = np.asarray(K) * np.cos(phase)
    Y = np.asarray(K) * np.sin(phase)
    return X, Y


def directional_descriptor(Y: Array, epsilon0: float = 1.0e-15) -> float:
    """Network-level directional descriptor used only as a comparator."""
    Ys = (Y + Y.T) / 2.0
    Ya = (Y - Y.T) / 2.0
    return float(np.linalg.norm(Ya, "fro") / (np.linalg.norm(Ys, "fro") + epsilon0))


def oa_quotient_rhs(
    _t: float,
    x: Array,
    K: Array,
    beta: float | Array,
    Delta: float | Array = 0.0,
    omega: float | Array = 0.0,
) -> Array:
    """General M-population OA flow after quotienting the global phase."""
    K = np.asarray(K)
    M = K.shape[0]
    rho = np.asarray(x[:M])
    # Preserve complex dtype for complex-step differentiation.
    zero = 0.0 * x[0]
    phi = np.r_[x[M:], zero]
    Delta = np.broadcast_to(np.asarray(Delta), (M,))
    omega = np.broadcast_to(np.asarray(omega), (M,))
    phase = phi[None, :] - phi[:, None] + np.asarray(beta)
    sine_sum = np.sum(K * rho[None, :] * np.sin(phase), axis=1)
    cosine_sum = np.sum(K * rho[None, :] * np.cos(phase), axis=1)
    drho = -Delta * rho + (1.0 - rho * rho) * sine_sum / 2.0
    dphi = omega - (1.0 + rho * rho) * cosine_sum / (2.0 * rho)
    return np.r_[drho, dphi[:-1] - dphi[-1]]


def complex_step_jacobian(fun: Callable[[Array], Array], x: Array) -> Array:
    """Machine-precision Jacobian for an analytic real vector field."""
    x = np.asarray(x, float)
    h = 1.0e-28
    eye = np.eye(x.size)
    return np.column_stack(
        [np.imag(fun(x.astype(complex) + 1j * h * e)) / h for e in eye]
    )


def ring_coupling(
    M: int,
    A: float,
    eta: float,
    self_gradient: float = 0.0,
) -> Array:
    """Nearest-neighbor directed ring with normalized total cross weight.

    ``K[sigma,sigma+1]=nu(1+eta)/2`` and
    ``K[sigma,sigma-1]=nu(1-eta)/2``.  A small linear onsite gradient can
    be used to unfold the semisimple Fourier degeneracy of the perfectly
    circulant reciprocal ring.
    """
    mu, nu = couplings(A)
    onsite = 1.0 + self_gradient * np.linspace(-1.0, 1.0, M)
    K = np.zeros((M, M), float)
    for sigma in range(M):
        K[sigma, sigma] = mu * onsite[sigma]
        K[sigma, (sigma + 1) % M] = nu * (1.0 + eta) / 2.0
        K[sigma, (sigma - 1) % M] = nu * (1.0 - eta) / 2.0
    return K


@dataclass
class LockedRing:
    M: int
    A: float
    beta: float
    eta: float
    Delta: float
    self_gradient: float
    x: Array
    J: Array


def solve_locked_ring(
    M: int,
    A: float,
    beta: float,
    eta: float,
    Delta: float,
    self_gradient: float,
    guess: Array,
) -> LockedRing:
    """Solve and linearize the controlled M-population ring benchmark."""
    K = ring_coupling(M, A, eta, self_gradient)
    fun = lambda y: oa_quotient_rhs(0.0, y, K, beta, Delta)
    sol = root(fun, np.asarray(guess, float), tol=1.0e-12)
    residual = np.linalg.norm(fun(sol.x), ord=np.inf)
    if residual > 1.0e-9:
        raise RuntimeError(
            f"M={M} ring solve failed at eta={eta:g}; residual={residual:.3e}"
        )
    J = complex_step_jacobian(fun, sol.x)
    return LockedRing(M, A, beta, eta, Delta, self_gradient, sol.x, J)
