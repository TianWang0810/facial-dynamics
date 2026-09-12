# docs/ Directory Index

This directory contains non-code project knowledge -- engineering logs and
technical decisions -- kept separate from source code so both evolve
independently and remain easy to search.

## engineering_log/
Chronological problem to diagnosis to root-cause to fix records from each
validation round. One file per major validation session/environment.

- local_validation_day1.md -- macOS local validation (Observation and
  Geometry layers, 7 issues resolved)
- hpc_deployment.md -- AICR HPC deployment validation (2 issues resolved)

## decisions/
Standalone technical decision records (Architecture Decision Record style:
why we chose A over B), one file per decision so each remains easy to find
and update independently as the project evolves.

- mediapipe_version_pinning.md -- why mediapipe is pinned to 0.10.21
