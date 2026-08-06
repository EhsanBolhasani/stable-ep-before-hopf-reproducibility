# Independent local two-population diagnostics

`code/independent_local_diagnostics.py` independently regenerates the local
numerical evidence for the two-population Ott--Antonsen benchmark.  It does
not read manuscript CSV files and contains no hard-coded critical points,
periods, eigenvalues, or Floquet exponents.

Run from the package root:

```bash
python code/independent_local_diagnostics.py
```

The script writes `code/outputs_local/` with:

- `locked_branch_beta_0p10.csv`: adaptive continuation resolving the narrow
  saddle-node--EP interval and continuing to Hopf;
- `relaxation_pulses_beta_0p10.csv` and its summary: the declared small
  coherence pulse below, at, and above the EP;
- `transverse_floquet_cycles.csv`: period, the nontrivial in-boundary Floquet
  exponent and multiplier, the boundary-normal exponent and multiplier, and
  the autonomous trivial multiplier;
- `homoclinic_periods_beta_0p06.csv` and
  `homoclinic_fit_windows_beta_0p06.csv`: an explicitly labeled auxiliary
  cross-cut validation retained for provenance; and
- `summary.json` plus figure-ready diagnostic PDFs and PNGs.

The main global result is not taken from the auxiliary `beta=0.06` tables.
`code/same_cut_homoclinic.py` independently shoots the stable and unstable
manifolds and continues the breathing periods on the paper's main
`beta=0.10` cut.  `code/full_floquet_validation.py` separately integrates the
monodromy equation and checks the Liouville multiplier for all sampled cycles.

The pulse norm and transient gain use the declared Euclidean metric in
`(r,psi)` coordinates.  Their magnitudes are not invariant under arbitrary
coordinate rescaling.  The discriminant sign, node/Jordan/focus response
classes, and the Floquet multipliers of the declared quotient dynamics are
the coordinate-independent numerical claims.
