# A stable exceptional point before chimera breathing: reproducibility repository

## September 2026 revision supplement

This local supplementary distribution adds numerical refinement and metric
checks and changes the Fig. 3 growth-rate label from mu_J to gamma_J.
It was prepared from GitHub commit
`005bdef0869502cc5e43caeefd347932ae5549b2`.
This public revision distribution corresponds to the revised v1.1.0
release, archived at
[10.5281/zenodo.22861086](https://doi.org/10.5281/zenodo.22861086).
The baseline v1.0.0 release remains available as a historical archive at
[10.5281/zenodo.21828808](https://doi.org/10.5281/zenodo.21828808).


The complete revised Online Resource 1 is included in `online_resource_1/`.
Sections S1–S4 retain and clarify the original derivations and validation;
S5–S9 contain the additional revision analyses and source map.
`data/revision/` contains their actual outputs. Regenerate these checks with:

```bash
python -m pip install -r requirements_revision.txt
python code/revision_checks.py --repository . --output data/revision
```

The new calculations were executed with Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, and mpmath 1.4.1. The pinned original environment below remains
the environment record for the pre-existing analyses. The homoclinic
refinement and finite-population tables were inspected, not regenerated in
the revision check. A refreshed manifest describes this local distribution.

To regenerate the clean and blue-marked Fig. 3 from the existing data:

```bash
python -c "import sys; sys.path.insert(0, 'code'); import make_figures as f; f.figure3_hopf_floquet(); f.figure3_hopf_floquet(mark_changes=True)"
```

The blue figure marks only the changed growth-rate labels. A LaTeX
installation is required for the authoritative vector render.

This repository contains the Python code, numerical data, and figure-building
scripts for the manuscript:

**A stable exceptional point reorganizes relaxation before the Hopf
bifurcation to a breathing chimera**

Authors: Ehsan Bolhasani, Seyed Hamed Aboutalebi, and Matjaž Perc.

The repository is intended to reproduce the numerical analyses and figures
from the archived data tables. It does not include the journal submission
manuscript, cover letter, or editorial files.

## Contents

- `code/`: model definitions, continuation, stability, pulse-response,
  homoclinic, finite-size, validation, and plotting scripts.
- `data/`: machine-readable numerical tables and JSON summaries.
- `figures/`: publication figures in PDF and PNG form.
- `reports/`: supporting diagnostic figures and audit notes.
- `online_resource_1/`: complete revised supplementary PDF, LaTeX, and figures.
- `SOURCE_MAP.csv`: submission labels and their corresponding figure sources.
- `run_all.py`: top-level reproduction driver.
- `requirements.txt`: pinned Python dependencies.
- `MANIFEST.sha256`: release integrity manifest.

## Python Setup

The tested Python environment is CPython 3.12 with pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, a typical PowerShell setup is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Integrity Check

After downloading and extracting the repository, verify that the release files
match the archived SHA-256 manifest:

```bash
python code/release_manifest.py verify
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe code\release_manifest.py verify
```

This manifest checks the scientific code, data, figures, reports, and metadata.
Git housekeeping files such as `.gitignore` and `.gitattributes` are ignored so
that GitHub source archives and manually downloaded ZIP files verify in the same
way.

## Quick Check

To run a Python-only smoke check without rebuilding figures or compiling
LaTeX artifacts:

```bash
python run_all.py --mode smoke --bootstrap 20 --skip-figures --skip-latex
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run_all.py --mode smoke --bootstrap 20 --skip-figures --skip-latex
```

## Reproducing the Reported Data Products

The default paper-mode workflow rebuilds the reported summaries and figures
from the archived finite-size raw tables. Because journal source files are not
distributed here, LaTeX compilation is skipped automatically:

```bash
python run_all.py --mode paper
```

The expensive full workflow additionally regenerates the finite-size raw
oscillator ensembles:

```bash
python run_all.py --mode full --workers 4 --skip-latex
```

The full workflow may require hours depending on CPU count and vectorized math
performance.

## Notes

The finite-size raw tables distributed in `data/` are retained so that the
reported figures and scaling summaries can be reproduced without rerunning the
most expensive simulations. Random seeds used for finite-size simulations,
bootstrap resampling, pulse-noise checks, and microscopic-reset checks are
fixed in the source code.

## Citation

Please cite the associated manuscript and identify the version used.
For this revised release, please cite the version-specific DOI:
https://doi.org/10.5281/zenodo.22861086.
The baseline v1.0.0 DOI remains available for the original release:
https://doi.org/10.5281/zenodo.21828808.

## Compiling the supplementary document

From `online_resource_1/`, run this command three times:

```bash
pdflatex -interaction=nonstopmode -halt-on-error supplement_clean.tex
```

Its figure paths are local. The full journal submission package separately contains
both manuscript variants, both supplementary variants, and a `build_pdfs.py`
helper. The public data-reproduction driver does not compile these journal files.
