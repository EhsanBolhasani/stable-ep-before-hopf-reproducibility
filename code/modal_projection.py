#!/usr/bin/env python3
"""Metric-aware two-mode projections for ``2M-1`` dimensional OA systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import schur
from scipy.optimize import root_scalar

from model import (
    LockedRing,
    solve_locked_ring,
    two_by_two_decomposition,
)


Array = np.ndarray


def natural_quotient_metric(rho: Array) -> Array:
    """Metric induced by ``sum_sigma |dz_sigma|^2`` after phase quotient.

    The amplitude block is the identity.  If population ``M`` is the phase
    reference and ``w_sigma=rho_sigma**2``, the phase block is

        diag(w_1,...,w_{M-1}) - w w^T / sum_sigma w_sigma.
    """
    rho = np.asarray(rho, float)
    M = rho.size
    w = rho * rho
    phase = np.diag(w[:-1]) - np.outer(w[:-1], w[:-1]) / np.sum(w)
    W = np.zeros((2 * M - 1, 2 * M - 1), float)
    W[:M, :M] = np.eye(M)
    W[M:, M:] = phase
    return W


def metric_similarity(J: Array, W: Array) -> tuple[Array, Array]:
    """Return the Euclidean representation in coordinates set by metric W.

    ``W=C.T@C`` and ``J_metric=C@J@inv(C)``.  Similarity preserves all
    eigenvalues, discriminants, and Jordan defects; it only declares the
    inner product used to split symmetric and skew-adjoint response.
    """
    C = np.linalg.cholesky(W).T
    J_metric = C @ J @ np.linalg.inv(C)
    return J_metric, C


def align_plane(Q: Array, previous_Q: Array | None) -> Array:
    """Orthogonal Procrustes alignment of consecutive Schur frames."""
    if previous_Q is None:
        return Q
    U, _singular, Vt = np.linalg.svd(Q.T @ previous_Q)
    return Q @ (U @ Vt)


@dataclass
class ModalBlock:
    B: Array
    Q: Array
    eigenvalues: Array
    spectral_gap: float
    invariance_residual: float
    diagnostics: dict[str, float]


def rightmost_real_schur_block(
    J: Array,
    previous_Q: Array | None = None,
) -> ModalBlock:
    """Extract the isolated rightmost two-mode real Schur block.

    This controlled benchmark has a rightmost pair separated from all other
    eigenvalues.  The selection threshold is placed midway between the
    second- and third-rightmost real parts.  Production use on a general
    branch should select by spectral-projector overlap; the reported overlap
    and spectral-gap checks make the assumption testable here.
    """
    eig = np.linalg.eigvals(J)
    order = np.argsort(eig.real)
    target = eig[order[-2:]]
    rest = eig[order[:-2]]
    threshold = (np.min(target.real) + np.max(rest.real)) / 2.0

    def choose(wr: float, _wi: float | None = None) -> bool:
        return bool(wr > threshold)

    T, Z, selected = schur(J, output="real", sort=choose)
    if selected != 2:
        raise RuntimeError(f"ordered real Schur selected {selected} modes, expected 2")
    Q = align_plane(Z[:, :2], previous_Q)
    B = Q.T @ J @ Q
    block_eig = np.linalg.eigvals(B)
    gap = float(min(abs(a - b) for a in block_eig for b in rest))
    residual = float(np.linalg.norm(J @ Q - Q @ B, ord="fro"))
    return ModalBlock(
        B=B,
        Q=Q,
        eigenvalues=block_eig,
        spectral_gap=gap,
        invariance_residual=residual,
        diagnostics=two_by_two_decomposition(B),
    )


def initial_ring_guess(M: int, beta: float, Delta: float) -> Array:
    """Uniform-state approximation used only to initialize Newton solves."""
    r2 = max(0.05, 1.0 - 2.0 * Delta / np.sin(beta))
    return np.r_[np.full(M, np.sqrt(r2)), np.zeros(M - 1)]


def ring_modal_point(
    M: int,
    eta: float,
    guess: Array,
    previous_Q: Array | None = None,
    A: float = 0.25,
    beta: float = 0.28,
    Delta: float = 0.005,
    self_gradient: float = 0.02,
) -> tuple[LockedRing, ModalBlock]:
    """Solve the ring and extract its critical physical-metric block."""
    locked = solve_locked_ring(
        M, A, beta, eta, Delta, self_gradient, np.asarray(guess, float)
    )
    W = natural_quotient_metric(locked.x[:M])
    J_metric, _C = metric_similarity(locked.J, W)
    block = rightmost_real_schur_block(J_metric, previous_Q)
    return locked, block


def continue_ring_modal(
    M: int,
    etas: Array,
    A: float = 0.25,
    beta: float = 0.28,
    Delta: float = 0.005,
    self_gradient: float = 0.02,
) -> list[dict[str, object]]:
    """Continue the controlled M-population locked branch in directionality."""
    guess = initial_ring_guess(M, beta, Delta)
    Q = None
    out: list[dict[str, object]] = []
    for eta in np.asarray(etas, float):
        locked, block = ring_modal_point(
            M, eta, guess, Q, A, beta, Delta, self_gradient
        )
        guess, Q = locked.x, block.Q
        full_eig = np.linalg.eigvals(locked.J)
        out.append(
            {
                "M": M,
                "eta": float(eta),
                "x": locked.x.copy(),
                "B": block.B.copy(),
                "Q": block.Q.copy(),
                "eigenvalues": block.eigenvalues.copy(),
                "spectral_gap": block.spectral_gap,
                "invariance_residual": block.invariance_residual,
                "spectral_abscissa": float(np.max(full_eig.real)),
                **block.diagnostics,
            }
        )
    return out


def locate_ring_ep(
    M: int,
    bracket: tuple[float, float] = (0.15, 0.25),
    A: float = 0.25,
    beta: float = 0.28,
    Delta: float = 0.005,
    self_gradient: float = 0.02,
) -> dict[str, object]:
    """Locate and verify the first stable modal EP of the ring benchmark."""
    guess0 = initial_ring_guess(M, beta, Delta)
    cache: dict[float, Array] = {}

    def point(eta: float) -> tuple[LockedRing, ModalBlock]:
        if cache:
            key = min(cache, key=lambda z: abs(z - eta))
            guess = cache[key]
        else:
            guess = guess0
        locked, block = ring_modal_point(
            M, eta, guess, None, A, beta, Delta, self_gradient
        )
        cache[float(eta)] = locked.x
        return locked, block

    solution = root_scalar(
        lambda eta: point(float(eta))[1].diagnostics["discriminant"],
        bracket=bracket,
        method="brentq",
        xtol=2.0e-13,
    )
    eta_ep = float(solution.root)
    locked, block = point(eta_ep)
    B = block.B
    lam = float(np.trace(B) / 2.0)
    singular = np.linalg.svd(B - lam * np.eye(2), compute_uv=False)
    h = 2.0e-5
    dD = (
        point(eta_ep + h)[1].diagnostics["discriminant"]
        - point(eta_ep - h)[1].diagnostics["discriminant"]
    ) / (2.0 * h)
    return {
        "M": M,
        "eta_EP": eta_ep,
        "x": locked.x,
        "B": B,
        "lambda_star": lam,
        "singular_values_B_minus_lambdaI": singular,
        "dD_deta": float(dD),
        "spectral_gap": block.spectral_gap,
        "invariance_residual": block.invariance_residual,
        "spectral_abscissa": float(np.max(np.linalg.eigvals(locked.J).real)),
        **block.diagnostics,
    }
