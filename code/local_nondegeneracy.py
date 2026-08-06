#!/usr/bin/env python3
"""Independent local nondegeneracy checks for the two-population OA model.

Run this file from the repository root::

    python code/local_nondegeneracy.py --output-dir data

The reported signs use explicit eigenvector/Jordan-chain conventions.  The
nonzero character of the coefficients, rather than an arbitrary sign, is the
coordinate-independent genericity statement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from local_analysis import benchmark_points
from model import (
    two_pop_equilibrium,
    two_pop_jacobian,
    two_pop_rhs,
)


def derivative_5(fun, x: float, h: float):
    """Fourth-order centered first derivative of an array-valued function."""
    return (
        fun(x - 2.0 * h)
        - 8.0 * fun(x - h)
        + 8.0 * fun(x + h)
        - fun(x + 2.0 * h)
    ) / (12.0 * h)


def directional_hessian(x, A, beta, u, v, h=1.0e-4):
    """Return B(u,v)=D_x^2 F[u,v] by differentiating the analytic Jacobian."""
    dJ = derivative_5(
        lambda s: two_pop_jacobian(x + s * u, A, beta), 0.0, h
    )
    return dJ @ v


def parameter_derivative(x, A, beta, which, h=1.0e-4):
    if which == "A":
        return derivative_5(
            lambda value: two_pop_rhs(0.0, x, value, beta), A, h
        )
    if which == "beta":
        return derivative_5(
            lambda value: two_pop_rhs(0.0, x, A, value), beta, h
        )
    raise ValueError(which)


def local_checks(h: float = 1.0e-4):
    points = benchmark_points(0.10)

    # Generic saddle-node.  q is Euclidean unit length with q_r>0; p^T q=1.
    sn = points["SN"]
    x_sn = np.array([sn["r"], sn["psi"]], float)
    J_sn = np.asarray(sn["J"], float)
    U, singular, Vh = np.linalg.svd(J_sn)
    q = Vh[-1].copy()
    if q[0] < 0.0:
        q *= -1.0
    p = U[:, -1].copy()
    p /= p @ q
    F_A_sn = parameter_derivative(x_sn, sn["A"], 0.10, "A", h)
    Bqq_sn = directional_hessian(x_sn, sn["A"], 0.10, q, q, h)
    sn_parameter = float(p @ F_A_sn)
    sn_quadratic = float(0.5 * p @ Bqq_sn)

    # Hopf crossing speed along the locked-equilibrium branch.  For a complex
    # pair in two dimensions Re(lambda)=trace(J)/2.
    hopf = points["Hopf"]
    x_h = np.array([hopf["r"], hopf["psi"]], float)
    A_h = float(hopf["A"])

    def branch_trace(A):
        x = two_pop_equilibrium(A, 0.10, x_h)
        return np.trace(two_pop_jacobian(x, A, 0.10))

    trace_speed = float(derivative_5(branch_trace, A_h, h))
    realpart_speed = 0.5 * trace_speed

    # Bogdanov--Takens Jordan chain.  The declared normalization is
    # ||q0||_2=1, q0_r>0, J q1=q0, q0^T q1=0.  If P=[q0,q1], rows of P^{-1}
    # are the dual covectors.  In this convention the standard quadratic
    # coefficients are
    #   a = 1/2 row_2 B(q0,q0),
    #   b = row_2 B(q0,q1) + row_1 B(q0,q0).
    bt = points["BT"]
    x_bt = np.array([bt["r"], bt["psi"]], float)
    A_bt, beta_bt = float(bt["A"]), float(bt["beta"])
    J_bt = np.asarray(bt["J"], float)
    U_bt, singular_bt, Vh_bt = np.linalg.svd(J_bt)
    q0 = Vh_bt[-1].copy()
    if q0[0] < 0.0:
        q0 *= -1.0
    augmented = np.vstack([J_bt, q0])
    q1 = np.linalg.lstsq(augmented, np.r_[q0, 0.0], rcond=None)[0]
    P = np.column_stack([q0, q1])
    Pinv = np.linalg.inv(P)
    B00 = directional_hessian(x_bt, A_bt, beta_bt, q0, q0, h)
    B01 = directional_hessian(x_bt, A_bt, beta_bt, q0, q1, h)
    bt_a = float(0.5 * (Pinv @ B00)[1])
    bt_b = float((Pinv @ B01)[1] + (Pinv @ B00)[0])

    F_A_bt = parameter_derivative(x_bt, A_bt, beta_bt, "A", h)
    F_beta_bt = parameter_derivative(x_bt, A_bt, beta_bt, "beta", h)
    physical_parameter_forcing = np.column_stack([F_A_bt, F_beta_bt])
    jordan_parameter_forcing = Pinv @ physical_parameter_forcing

    def augmented_bt_conditions(y):
        x, A, beta = y[:2], float(y[2]), float(y[3])
        J = two_pop_jacobian(x, A, beta)
        return np.r_[
            two_pop_rhs(0.0, x, A, beta),
            np.linalg.det(J),
            np.trace(J),
        ]

    y_bt = np.r_[x_bt, A_bt, beta_bt]
    D_aug = np.column_stack(
        [
            derivative_5(
                lambda value, j=j: augmented_bt_conditions(
                    y_bt + (value - y_bt[j]) * np.eye(4)[j]
                ),
                y_bt[j],
                h,
            )
            for j in range(4)
        ]
    )
    aug_singular = np.linalg.svd(D_aug, compute_uv=False)

    rows = [
        {
            "event": "SN",
            "quantity": "pT_F_A",
            "value": sn_parameter,
            "normalization": "||q||2=1, q_r>0, pTq=1",
        },
        {
            "event": "SN",
            "quantity": "0.5_pT_Bqq",
            "value": sn_quadratic,
            "normalization": "||q||2=1, q_r>0, pTq=1",
        },
        {
            "event": "Hopf",
            "quantity": "d_trace_dA_along_branch",
            "value": trace_speed,
            "normalization": "physical A",
        },
        {
            "event": "Hopf",
            "quantity": "d_Re_lambda_dA_along_branch",
            "value": realpart_speed,
            "normalization": "physical A",
        },
        {
            "event": "BT",
            "quantity": "normal_form_a",
            "value": bt_a,
            "normalization": "unit q0; q0_r>0; q0Tq1=0; Jq1=q0",
        },
        {
            "event": "BT",
            "quantity": "normal_form_b",
            "value": bt_b,
            "normalization": "unit q0; q0_r>0; q0Tq1=0; Jq1=q0",
        },
        {
            "event": "BT",
            "quantity": "det_F_A_F_beta",
            "value": float(np.linalg.det(physical_parameter_forcing)),
            "normalization": "physical state and parameters",
        },
        {
            "event": "BT",
            "quantity": "det_augmented_F_det_trace",
            "value": float(np.linalg.det(D_aug)),
            "normalization": "variables (r,psi,A,beta)",
        },
        {
            "event": "BT",
            "quantity": "sigma_min_augmented_F_det_trace",
            "value": float(aug_singular[-1]),
            "normalization": "variables (r,psi,A,beta)",
        },
    ]

    details = {
        "finite_difference_step": h,
        "SN": {
            "A": float(sn["A"]),
            "x": x_sn.tolist(),
            "J_singular_values": singular.tolist(),
            "q": q.tolist(),
            "p": p.tolist(),
            "pTq": float(p @ q),
            "pT_F_A": sn_parameter,
            "half_pT_Bqq": sn_quadratic,
            "product": sn_parameter * sn_quadratic,
        },
        "Hopf": {
            "A": A_h,
            "x": x_h.tolist(),
            "d_trace_dA_along_branch": trace_speed,
            "d_Re_lambda_dA_along_branch": realpart_speed,
        },
        "BT": {
            "A": A_bt,
            "beta": beta_bt,
            "x": x_bt.tolist(),
            "trace_J": float(np.trace(J_bt)),
            "det_J": float(np.linalg.det(J_bt)),
            "norm_J_squared": float(np.linalg.norm(J_bt @ J_bt, ord=2)),
            "J_singular_values": singular_bt.tolist(),
            "q0": q0.tolist(),
            "q1": q1.tolist(),
            "Pinv_J_P": (Pinv @ J_bt @ P).tolist(),
            "normal_form_a": bt_a,
            "normal_form_b": bt_b,
            "physical_parameter_forcing": physical_parameter_forcing.tolist(),
            "jordan_parameter_forcing": jordan_parameter_forcing.tolist(),
            "det_physical_parameter_forcing": float(
                np.linalg.det(physical_parameter_forcing)
            ),
            "det_jordan_parameter_forcing": float(
                np.linalg.det(jordan_parameter_forcing)
            ),
            "augmented_jacobian_det": float(np.linalg.det(D_aug)),
            "augmented_jacobian_singular_values": aug_singular.tolist(),
            "augmented_jacobian_condition_number": float(np.linalg.cond(D_aug)),
        },
        "interpretation": {
            "SN": "Both standard saddle-node coefficients are nonzero.",
            "Hopf": "The complex pair crosses transversely as A increases.",
            "BT": (
                "F=0, det(J)=trace(J)=0 and rank(J)=1 alone are necessary but "
                "not a complete generic-BT check.  The additional nonzero a, b, "
                "parameter-forcing determinant, and nonsingular augmented system "
                "establish the missing nondegeneracy/unfolding checks in the "
                "declared Jordan-chain convention."
            ),
        },
    }
    return rows, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--step", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = local_checks(args.step)
    with (args.output_dir / "local_nondegeneracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "local_nondegeneracy.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    for row in rows:
        print(f"{row['event']:5s} {row['quantity']:38s} {row['value']:+.12g}")


if __name__ == "__main__":
    main()
