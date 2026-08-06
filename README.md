# A stable exceptional point before chimera breathing: reproducibility repository

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

If you use this code or data, please cite the associated manuscript and the
archived repository DOI once available.
