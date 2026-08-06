#!/usr/bin/env python3
"""Reproduce the numerical analyses and figures associated with the manuscript.

``paper`` mode (the default) rebuilds reported summaries and presentation
artifacts from the archived finite-N raw tables.  ``full`` mode additionally
regenerates both expensive oscillator-level ensembles with the published
settings.  ``smoke`` mode reuses the archived long-running local, homoclinic,
and Floquet tables while exercising downstream analysis, plotting, validation,
and (unless disabled) LaTeX compilation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MANUSCRIPT = ROOT / "manuscript"


def run(
    command: list[str],
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> float:
    print("+", " ".join(command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    elapsed = time.perf_counter() - start
    print(f"  completed in {elapsed:.1f} s", flush=True)
    return elapsed


def tool_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.splitlines()[0].strip() if result.stdout else "available"


def record_environment(mode: str) -> None:
    import matplotlib
    import numpy
    import scipy

    report = {
        "workflow_mode": mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "latexmk": tool_version(["latexmk", "-version"]),
        "bibtex": tool_version(["bibtex", "--version"]),
        "pdftocairo": tool_version(["pdftocairo", "-v"]),
    }
    (DATA / "reproduction_environment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Environment:", json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the paper from archived raw tables, or explicitly "
            "regenerate the expensive finite-N ensembles."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "paper", "full"),
        default="paper",
        help=(
            "smoke: reuse long-running archived tables; paper: recompute all "
            "reported analyses except finite-N raw ensembles (default); full: "
            "also regenerate both finite-N raw ensembles"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help=(
            "stratified bootstrap replicates (default: 100 in smoke mode, "
            "10000 otherwise)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="worker processes for full finite-N regeneration",
    )
    parser.add_argument(
        "--skip-local-audit",
        action="store_true",
        help="reuse the distributed local audit tables",
    )
    parser.add_argument(
        "--skip-global-shooting",
        action="store_true",
        help="reuse the distributed homoclinic shooting and period tables",
    )
    parser.add_argument(
        "--skip-full-floquet",
        action="store_true",
        help="reuse the distributed independent full-spectrum Floquet table",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "reuse the distributed publication figures; useful for a "
            "Python-only dependency check without a LaTeX installation"
        ),
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help=(
            "do not compile manuscript PDFs; this is automatic when the "
            "optional manuscript source directory is absent"
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="do not run the tolerance-based scientific/artifact checks",
    )
    parser.add_argument(
        "--write-release-manifest",
        action="store_true",
        help=(
            "after a successful run, freeze the current curated tree in "
            "MANIFEST.sha256; use only for a final release"
        ),
    )
    args = parser.parse_args()

    if args.bootstrap is not None and args.bootstrap < 20:
        parser.error("--bootstrap must be at least 20")
    if args.workers < 1:
        parser.error("--workers must be positive")
    bootstrap = (
        args.bootstrap
        if args.bootstrap is not None
        else (100 if args.mode == "smoke" else 10000)
    )

    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
    # ``record_environment`` imports Matplotlib in this process, whereas the
    # other commands receive the copied environment below.
    os.environ.setdefault("MPLBACKEND", env["MPLBACKEND"])
    os.environ.setdefault("MPLCONFIGDIR", env["MPLCONFIGDIR"])
    local_texmf = ROOT / "texmf"
    if local_texmf.exists():
        env["TEXINPUTS"] = (
            str(local_texmf / "tex" / "latex")
            + "//"
            + os.pathsep
            + env.get("TEXINPUTS", "")
        )
        env["BSTINPUTS"] = (
            str(local_texmf / "bibtex" / "bst")
            + "//"
            + os.pathsep
            + env.get("BSTINPUTS", "")
        )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    record_environment(args.mode)

    smoke = args.mode == "smoke"
    skip_local = args.skip_local_audit or smoke
    skip_shooting = args.skip_global_shooting or smoke
    skip_floquet = args.skip_full_floquet or smoke
    if args.mode == "full":
        print(
            "Full mode selected: regenerating the published finite-N raw "
            "ensembles. This is the expensive workflow.",
            flush=True,
        )
        for dt, seeds in ((0.04, 8), (0.02, 3)):
            run(
                [
                    sys.executable,
                    str(CODE / "finite_n_true_quotient.py"),
                    "--dt",
                    str(dt),
                    "--seeds",
                    str(seeds),
                    "--workers",
                    str(args.workers),
                    "--T",
                    "800",
                    "--discard",
                    "250",
                ],
                env=env,
            )
    else:
        print(
            f"{args.mode.capitalize()} mode uses the distributed finite-N raw "
            "tables; select --mode full for deterministic raw regeneration.",
            flush=True,
        )

    if not skip_local:
        run([sys.executable, str(CODE / "independent_local_diagnostics.py")], env=env)
        source = CODE / "outputs_local"
        for name in (
            "locked_branch_beta_0p10.csv",
            "relaxation_pulses_beta_0p10.csv",
            "relaxation_pulse_summary_beta_0p10.csv",
            "transverse_floquet_cycles.csv",
            "homoclinic_periods_beta_0p06.csv",
            "homoclinic_fit_windows_beta_0p06.csv",
        ):
            shutil.copy2(source / name, DATA / name)

    if not skip_shooting:
        run(
            [
                sys.executable,
                str(CODE / "same_cut_homoclinic.py"),
                "--output-dir",
                str(DATA),
            ],
            env=env,
        )

    if not skip_floquet:
        run([sys.executable, str(CODE / "full_floquet_validation.py")], env=env)

    run([sys.executable, str(CODE / "dispersive_robustness.py")], env=env)
    run(
        [
            sys.executable,
            str(CODE / "finite_n_true_quotient.py"),
            "--summary-only",
            "--bootstrap",
            str(bootstrap),
        ],
        env=env,
    )
    run([sys.executable, str(CODE / "nonlinear_pulse_robustness.py")], env=env)
    run(
        [
            sys.executable,
            str(CODE / "local_nondegeneracy.py"),
            "--output-dir",
            str(DATA),
        ],
        env=env,
    )
    run(
        [
            sys.executable,
            str(CODE / "prepare_data.py"),
            "--bootstrap",
            str(bootstrap),
        ],
        env=env,
    )
    run([sys.executable, str(CODE / "ring_modal_counterexample.py")], env=env)

    if args.mode == "full":
        cache = DATA / "figure_cycle_cache.npz"
        if cache.exists():
            cache.unlink()
            print(f"Removed generated cycle cache before full rebuild: {cache}")
    if not args.skip_figures:
        run([sys.executable, str(CODE / "make_figures.py")], env=env)
    else:
        print(
            "Reusing the distributed publication figures (--skip-figures).",
            flush=True,
        )

    latex_available = MANUSCRIPT.is_dir()
    skip_latex = args.skip_latex or not latex_available
    if not latex_available and not args.skip_latex:
        print(
            "No manuscript source directory is distributed in this public "
            "repository; skipping LaTeX compilation.",
            flush=True,
        )

    if not skip_latex:
        if shutil.which("latexmk") is None:
            raise RuntimeError("latexmk is required to compile the manuscript PDFs")
        main_latexmk = [
            "latexmk",
            "-g",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ]
        run(main_latexmk, cwd=MANUSCRIPT, env=env)
        # The second forced pass makes a clean archive robust when a generated
        # bibliography is distributed but auxiliary files are omitted.
        run(main_latexmk, cwd=MANUSCRIPT, env=env)
        run(["latexmk", "-C", "supplement.tex"], cwd=MANUSCRIPT, env=env)
        run(
            [
                "latexmk",
                "-g",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "supplement.tex",
            ],
            cwd=MANUSCRIPT,
            env=env,
        )
        shutil.copy2(MANUSCRIPT / "supplement.pdf", MANUSCRIPT / "ESM_1.pdf")
        if (MANUSCRIPT / "cover_letter.tex").exists():
            run(
                [
                    "latexmk",
                    "-g",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "cover_letter.tex",
                ],
                cwd=MANUSCRIPT,
                env=env,
            )

    if not args.skip_validation:
        validation = [sys.executable, str(CODE / "validate_results.py")]
        if not skip_latex:
            validation.append("--check-latex")
        run(validation, env=env)

    if args.write_release_manifest:
        run(
            [sys.executable, str(CODE / "release_manifest.py"), "write"],
            env=env,
        )

    print(
        f"Reproduction completed successfully in {args.mode!r} mode. "
        f"Bootstrap replicates: {bootstrap}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
