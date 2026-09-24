# Measured experiment artifacts

`python -m experiments.profile` writes teacher observations to `profile/`.
`python -m experiments.evaluate` writes held-out observations, summaries and the
four requested CSVs to `latest/`, with copies of those CSVs in this directory.
Figures are regenerated from observations in `frontend/public/results/`.

Large run artifacts and downloaded models are ignored by Git. A compact real
demonstration run, when available, is retained under `demo/` with its provenance.
Unit-test fixtures are never published as research results.
