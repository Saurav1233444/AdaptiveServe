# AdaptiveServe implementation plan

**Goal:** Complete a reproducible adaptive inference research prototype.
**Architecture:** Native ONNX engine, Python learning/evaluation/API, React dashboard.
**Tech stack:** C++20, ONNX Runtime, PyTorch, FastAPI, React, Tailwind, Recharts.
**Spec:** `docs/design.md`; full user brief is authoritative.

## Global constraints

Real predictions and measurements only. Three ImageNet pretrained models.
Disjoint train/calibration/test data. CPU execution reproducible on a laptop.
No changes to shared services, no publication, no fabricated research claim.

## Tasks

- [x] Native engine: test registry errors and bounded scheduler first; replace fake
  workers with ONNX Runtime sessions, resource monitoring, JSON logs, pybind API,
  CMake dependency setup and CTest. Owner: native agent.
- [x] Learning: test feature ranges, data splits, routing constraints first; add
  dataset preparation, pretrained exports, CNN analyzer training, learned router,
  deterministic artifact metadata, Python tests. Owner: learning agent.
- [x] Dashboard: test upload and empty/error states first; implement all five pages,
  accessible responsive UI, chart data from API, interactive architecture; build
  and component tests. Owner: frontend agent.
- [x] Integration: test runtime, feedback DB, API failure states, aggregation first;
  add runtime, profile/train/benchmark orchestration, measured plots, dependencies,
  deployment and research documentation. Owner: root.
- [x] Run real end-to-end experiments, inspect artifacts, review all subsystems,
  address defects, capture measured results and final reproduction instructions.

The tasks use disjoint file ownership and the contracts in docs/design.md.
The workspace starts clean; work remains on feat/adaptiveserve-research in the
user's checkout so the complete result is immediately available there.
