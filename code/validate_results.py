#!/usr/bin/env python3
"""Tolerance-based validation of regenerated data and figure artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
MANUSCRIPT = ROOT / "manuscript"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def csv_rows(name: str) -> list[dict[str, str]]:
    path = DATA / name
    require(path.is_file(), f"missing data file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(check_latex: bool = False) -> dict[str, object]:
    checks: dict[str, object] = {}

    critical = {
        row["event"]: float(row["A"]) for row in csv_rows("critical_points.csv")
    }
    require(
        {"SN", "EP", "Hopf"}.issubset(critical),
        "critical_points.csv does not contain SN, EP, and Hopf rows",
    )
    hom = json.loads(
        (DATA / "same_cut_homoclinic_summary.json").read_text(encoding="utf-8")
    )
    A_hom = float(hom["raw_root"])
    require(
        critical["SN"] < critical["EP"] < critical["Hopf"] < A_hom,
        "required saddle-node < node-focus/EP < Hopf < homoclinic ordering failed",
    )
    checks["event_ordering"] = {
        "A_SN": critical["SN"],
        "A_EP": critical["EP"],
        "A_Hopf": critical["Hopf"],
        "A_homoclinic": A_hom,
    }

    declared_metric = csv_rows("locked_branch_declared_metric.csv")
    declared_fields = set(declared_metric[0])
    require(
        {
            "g_declared_metric",
            "gcrit_declared_metric",
            "R_B_declared_metric",
        }.issubset(declared_fields)
        and "N_Y" not in declared_fields
        and not any("physical_metric" in field for field in declared_fields),
        "declared-metric branch table retains a legacy field or lacks its "
        "renamed metric columns",
    )
    checks["declared_metric_branch"] = {
        "rows": len(declared_metric),
        "N_Y_removed": True,
    }

    solver_rows = csv_rows("homoclinic_solver_convergence.csv")
    expected_solver_settings = (
        (3.0e-9, 3.0e-11, 0.60),
        (3.0e-10, 3.0e-12, 0.30),
        (3.0e-11, 3.0e-13, 0.15),
        (3.0e-12, 3.0e-14, 0.075),
        (1.0e-12, 1.0e-14, 0.05),
    )
    require(
        len(solver_rows) == len(expected_solver_settings),
        "homoclinic solver table does not contain the five declared refinements",
    )
    for row, expected in zip(solver_rows, expected_solver_settings):
        actual = (
            float(row["ivp_rtol"]),
            float(row["ivp_atol"]),
            float(row["ivp_max_step"]),
        )
        require(
            np.allclose(actual, expected, rtol=1.0e-13, atol=0.0),
            f"unexpected homoclinic solver setting {actual}; expected {expected}",
        )
    solver_roots = np.asarray(
        [float(row["A_connection"]) for row in solver_rows], dtype=float
    )
    solver_span = float(np.ptp(solver_roots))
    declared_uncertainty = float(
        hom["conservative_empirical_parameter_uncertainty"]
    )
    section_span = float(hom["section_launch_root_span"])
    require(solver_span < 2.0e-10, "homoclinic solver sweep exceeds 2e-10")
    require(
        declared_uncertainty >= max(solver_span, section_span),
        "declared homoclinic sensitivity allowance is smaller than an observed sweep",
    )
    require(
        abs(float(hom["publication_value"]) - round(A_hom, 6)) < 5.0e-13,
        "homoclinic publication value is not the six-decimal rounding of raw_root",
    )
    require(
        abs(A_hom - 0.353696544) < 5.0e-10,
        "homoclinic root does not support the nine-decimal Supplement value",
    )
    require(
        abs(declared_uncertainty - 2.0e-10) < 1.0e-20,
        "homoclinic empirical uncertainty differs from the declared 2e-10",
    )
    bracket_splitting = np.asarray(
        hom["default_bracket_splitting_delta_psi"], dtype=float
    )
    require(
        bracket_splitting[0] < 0.0 < bracket_splitting[1],
        "homoclinic section splitting does not change sign on the declared bracket",
    )
    require(
        abs(float(hom["centered_d_splitting_dA"])) > 1.0e-3,
        "homoclinic section-splitting root is not numerically transverse",
    )
    robustness_rows = csv_rows("same_cut_homoclinic_manifold_robustness.csv")
    robustness_roots = np.asarray(
        [float(row["A_connection"]) for row in robustness_rows], dtype=float
    )
    measured_section_span = float(np.ptp(robustness_roots))
    require(
        len(robustness_rows) == 16 and measured_section_span < 3.0e-13,
        "homoclinic section/launch sweep does not support the Supplement claim",
    )
    require(
        abs(measured_section_span - section_span) < 1.0e-18,
        "homoclinic JSON section span disagrees with the underlying CSV",
    )
    checks["homoclinic_sensitivity"] = {
        "solver_span": solver_span,
        "section_launch_span": measured_section_span,
        "declared_empirical_uncertainty": declared_uncertainty,
        "number_of_solver_settings": len(solver_rows),
        "number_of_section_launch_settings": len(robustness_rows),
    }

    dispersive = csv_rows("dispersive_event_loci.csv")
    deltas = np.asarray([float(row["Delta"]) for row in dispersive])
    require(
        len(dispersive) == 21
        and np.allclose(deltas, np.linspace(0.0, 0.005, 21), atol=1.0e-15),
        "positive-dispersion table is not the declared 21-point uniform sweep",
    )
    for row in dispersive:
        require(
            float(row["A_SN"]) < float(row["A_EP"]) < float(row["A_Hopf"]),
            f"dispersive event ordering failed at Delta={row['Delta']}",
        )
        require(
            float(row["spectral_abscissa_EP"]) < 0.0,
            f"dispersive EP is not stable at Delta={row['Delta']}",
        )
        require(
            float(row["sv3_shifted"]) < 1.0e-10
            and float(row["sv2_shifted"]) > 1.0e-6,
            f"shifted-Jacobian rank-one nullspace check failed at "
            f"Delta={row['Delta']}",
        )
        require(
            max(
                float(row["residual_SN"]),
                float(row["residual_EP"]),
                float(row["residual_Hopf"]),
            )
            < 1.0e-7,
            f"dispersive solve residual too large at Delta={row['Delta']}",
        )
    row_001 = dispersive[int(np.argmin(np.abs(deltas - 0.001)))]
    require(
        abs(float(row_001["Delta"]) - 0.001) < 1.0e-15,
        "positive-dispersion table has no Delta=0.001 benchmark",
    )
    printed_001 = {
        "A_SN": 0.180443,
        "A_EP": 0.180620,
        "A_Hopf": 0.286121,
        "lambda_EP": -0.024284,
        "lambda_other_EP": -0.162539,
        "sv1_shifted": 0.215642,
        "sv2_shifted": 0.076519,
    }
    for field, printed in printed_001.items():
        require(
            abs(float(row_001[field]) - printed) <= 5.01e-7,
            f"Delta=0.001 field {field} does not support the manuscript rounding",
        )
    require(
        abs(float(row_001["sv3_shifted"])) < 5.0e-16,
        "Delta=0.001 smallest shifted-Jacobian singular value is inconsistent "
        "with the reported 3.8e-17",
    )
    pulse_rows = csv_rows("dispersive_pulse_responses.csv")
    reset_rows = csv_rows("microscopic_pulse_validation.csv")
    reset_sizes = sorted({int(row["N"]) for row in reset_rows})
    require(
        len(pulse_rows) == 3 * 1501
        and {row["case"] for row in pulse_rows} == {"node", "EP", "focus"}
        and all(abs(float(row["Delta"]) - 0.001) < 1.0e-15 for row in pulse_rows),
        "positive-dispersion pulse table does not match its declared protocol",
    )
    require(
        reset_sizes == [128, 256, 512, 1024, 2048, 4096]
        and all(
            sum(int(row["N"]) == size for row in reset_rows) == 12
            for size in reset_sizes
        ),
        "microscopic reset table does not contain 12 samples at each declared N",
    )
    checks["dispersive_continuation"] = {
        "number_of_Delta_values": len(dispersive),
        "Delta_min": float(deltas.min()),
        "Delta_max": float(deltas.max()),
        "pulse_rows": len(pulse_rows),
        "microscopic_reset_rows": len(reset_rows),
    }

    finite = json.loads(
        (DATA / "finite_n_true_quotient_fit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    for label in ("dt_0p04", "dt_0p02"):
        slope = float(finite[label]["true_chordal_quotient"]["slope"])
        require(-0.65 < slope < -0.35, f"unexpected finite-N slope {slope:g}")
    metric_check = finite["quotient_formula_validation"]
    require(
        int(metric_check["failures"]) == 0,
        "closed-form quotient-distance validation reported a failure",
    )
    require(
        float(metric_check["maximum_absolute_squared_distance_gap"])
        <= float(metric_check["squared_distance_tolerance"]),
        "closed-form quotient-distance gap exceeds its declared tolerance",
    )
    checks["finite_N"] = {
        "slope_dt_0p04": finite["dt_0p04"]["true_chordal_quotient"]["slope"],
        "slope_dt_0p02": finite["dt_0p02"]["true_chordal_quotient"]["slope"],
        "metric_formula_max_gap": metric_check[
            "maximum_absolute_squared_distance_gap"
        ],
    }

    floquet_path = DATA / "full_floquet_cycles_independent.csv"
    if floquet_path.is_file():
        floquet = csv_rows(floquet_path.name)
        require(
            all(float(row["inplane_exponent"]) < 0.0 for row in floquet),
            "a reported in-plane nontrivial Floquet exponent is nonnegative",
        )
        require(
            all(float(row["transverse_exponent"]) < 0.0 for row in floquet),
            "a reported transverse Floquet exponent is nonnegative",
        )
        require(
            max(float(row["trivial_multiplier_abs_error"]) for row in floquet)
            < 5.0e-6,
            "Floquet trivial-multiplier check exceeds 5e-6",
        )
        checks["Floquet"] = {
            "number_of_cycles": len(floquet),
            "maximum_trivial_multiplier_error": max(
                float(row["trivial_multiplier_abs_error"]) for row in floquet
            ),
        }

    figure_stems = [
        "Graphical_Abstract",
        *[f"Fig{index}_{suffix}" for index, suffix in (
            (1, "overview"),
            (2, "EP_and_relaxation"),
            (3, "Hopf_and_Floquet"),
            (4, "homoclinic_evidence"),
            (5, "parameter_synthesis"),
            (6, "modal_projection_M4_M5"),
            (7, "finite_N_validation"),
            (8, "dispersion_and_pulse"),
        )],
    ]
    required_figures = [
        f"{stem}.{extension}"
        for stem in figure_stems
        for extension in ("pdf", "png")
    ]
    missing_figures = [
        name for name in required_figures if not (FIGURES / name).is_file()
    ]
    require(not missing_figures, f"missing figure files: {missing_figures}")
    required_report_figures = (
        "nonlinear_pulse_robustness.pdf",
        "nonlinear_pulse_robustness.png",
    )
    missing_report_figures = [
        name for name in required_report_figures if not (REPORTS / name).is_file()
    ]
    require(
        not missing_report_figures,
        f"missing Supplementary Fig. S1 files listed in README: "
        f"{missing_report_figures}",
    )
    checks["figures"] = {
        "numbered_and_graphical_abstract_files": len(required_figures),
        "supplementary_S1_files": len(required_report_figures),
    }

    if MANUSCRIPT.is_dir():
        required_deliverables = (
            "main.tex",
            "main.pdf",
            "supplement.tex",
            "supplement.pdf",
            "ESM_1.pdf",
            "cover_letter.tex",
            "cover_letter.pdf",
            "sn-jnl.cls",
            "sn-mathphys-num.bst",
            "references.bib",
        )
        missing_deliverables = [
            name
            for name in required_deliverables
            if not (MANUSCRIPT / name).is_file()
        ]
        require(
            not missing_deliverables,
            f"missing manuscript deliverables: {missing_deliverables}",
        )
        checks["manuscript_deliverables"] = {
            "status": "validated",
            "required_files": len(required_deliverables),
        }
    else:
        require(
            not check_latex,
            "--check-latex requires the optional manuscript source directory",
        )
        checks["manuscript_deliverables"] = {"status": "not_distributed"}

    if check_latex:
        forbidden = (
            "LaTeX Error:",
            "undefined references",
            "undefined citations",
            "Citation `",
            "Reference `",
        )
        checked_logs = []
        for stem in ("main", "supplement", "cover_letter"):
            log = MANUSCRIPT / f"{stem}.log"
            if not log.is_file():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            for marker in forbidden:
                require(
                    marker.lower() not in text.lower(),
                    f"{log.name} contains {marker!r}",
                )
            checked_logs.append(log.name)
        require(checked_logs, "no LaTeX build logs were available for validation")
        checks["latex_logs"] = checked_logs

    result = {"status": "pass", "checks": checks}
    (DATA / "reproduction_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-latex",
        action="store_true",
        help="also reject LaTeX errors and unresolved citations/references",
    )
    args = parser.parse_args()
    print(json.dumps(validate(check_latex=args.check_latex), indent=2))


if __name__ == "__main__":
    main()
