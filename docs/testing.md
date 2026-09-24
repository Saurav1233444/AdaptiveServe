# Module inputs, expected behavior and verification

## Verified completion: 14 September 2026

| Exact command from repository root | Observed result |
|---|---|
| `.venv/bin/cmake --build build --parallel 2` | All native targets built successfully |
| `.venv/bin/ctest --test-dir build --output-on-failure` | 6/6 passed, including real ONNX identity execution |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q` | 39 passed; dependency deprecation warnings only |
| `npm --prefix frontend test -- --run` | 5 passed |
| `npm --prefix frontend run build` | TypeScript and Vite production build passed |
| `.venv/bin/python -m ruff check adaptiveserve experiments scripts tests` | All checks passed |
| `docker compose config --quiet` | Exit 0; container image execution was not tested |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/verify_demo.py` | Five real C++ policies, five dashboard routes and eight served figures passed |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.profile --threads 1 --warmup 3 --seed 42` | 1,200 teacher observations across 400 images |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/train.py --complexity-epochs 30 --router-epochs 200 --seed 42` | Both `.pt` checkpoints saved |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.evaluate --threads 1 --warmup 3 --repeats 3 --seed 42` | 1,500 requests; five summaries and requested CSVs |
| `.venv/bin/python -m experiments.plot --results results/latest --output frontend/public/results` | Eight nonempty figures from measured observations |

The API tests required execution outside the Codex sandbox because its network
restrictions stalled the asynchronous TestClient event loop. No application
workaround was applied. A connected browser was unavailable: component tests,
production build, direct page routes and real multipart uploads were verified,
but a browser visual review was not performed.

The copied environment had stale interpreter, editable-package and CMake paths.
Python 3.12 was restored under `.deps/python`, the editable package was reinstalled,
and CMake was reconfigured in this checkout. Local CMake and JSON headers live
in `.venv`; the existing build can be rebuilt using the commands above.
For a fresh machine, follow the standard setup below instead of copying `.venv`.

The real smoke check writes live feedback and refreshes `docs/smoke-results.json`.
Its first-load timings are not used for research comparisons. The retained
benchmark and interpretation are in [results.md](results.md).

Test fixtures may use small deterministic tensors or local synthetic images.
They exercise behavior and are never copied into `results/` or the dashboard.
Production inference always executes an exported neural network.

| Module | Sample input | Expected output / assertion | Tests |
|---|---|---|---|
| Dataset | Imagenette WNID `n01440764` | ImageNet label `0`; no repeated IDs/paths across splits | `test_data.py` |
| Features | Flat RGB image and seeded noisy image | Six finite normalized statistics; noise/entropy distinguish these inputs | `test_features.py` |
| CNN analyzer | RGB image with trained checkpoint | Finite score in `[0,1]`, 16-value embedding and measured analyzer milliseconds | `test_complexity.py` |
| Training | One training ID plus a held-out ID in observation CSV | Both checkpoints record only the manifest training ID | `test_training.py` |
| Export | All three pretrained weight specs | Correct transforms, 1,000 labels, ONNX/PyTorch parity | `test_export.py`, real export CLI |
| Rule router | Complexity `.2`, ample calibrated budgets | Prefer `mobilenet_v3_small`; `.4` prefers `resnet50`; `.7` prefers `efficientnet_b0` | `test_routing.py`, `test_router.cpp` |
| Learned router | Valid context and calibrated profiles | Exactly one feasible model; if none feasible return explicit violation | `test_routing.py` |
| Native registry | Duplicate names / missing ONNX path | Reject invalid metadata; permit absent file until actual inference | `test_registry.cpp`, `test_engine.cpp` |
| Native scheduler | Admission request beyond capacity | Explicit overload rejection; lease release frees capacity | `test_scheduler.cpp` |
| Native monitor/log | Process resource sample, escaped model name | Finite process RSS/CPU and valid escaped JSON | `test_monitor.cpp` |
| Native engine | Float32 identity fixture `[1,3,2,2]` | Output matches input, first load recorded, subsequent load zero, unload works | `test_onnx_integration.cpp` |
| Runtime | RGB `400×300` image | Contiguous normalized float32 `[1,3,224,224]`; missing artifact fails | `test_runtime.py` |
| Service | Analyzer chooses one model | Only that model executes; measured budget violations remain visible | `test_service.py` |
| Feedback | Concurrent insertions with distinct request IDs | All requests persist and arithmetic average is exact | `test_feedback.py` |
| API | Invalid PNG bytes; valid PNG; negative budget | HTTP 400; prediction/feedback; HTTP 422 | `test_api.py` |
| Evaluation | Two measured requests: 10 ms / 30 ms, one correct | Mean 20 ms; 50% accuracy; 50% misses at a 15 ms budget | `test_evaluation.py` |
| Plots | Complete observation CSV and summary | Exactly eight nonempty figures; no files means explicit failure | `test_plots.py` |
| Dashboard | Empty API then image upload | Empty states, preview, request, prediction/error and working navigation | `frontend/src/App.test.tsx` |

## Commands

```bash
uv sync --frozen --python 3.12 --extra dev
bash scripts/setup_native.sh
.venv/bin/python -m pytest -q
ctest --test-dir build --output-on-failure
.venv/bin/python -m ruff check adaptiveserve experiments scripts tests
cd frontend
npm ci
npm test -- --run
npm run build
```

The optional native identity fixture validates actual ONNX execution without
downloading large weights. To reproduce that additional CTest:

```bash
.venv/bin/python - <<'PY'
import onnx
from onnx import TensorProto, helper
input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,3,2,2])
output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,3,2,2])
graph = helper.make_graph([helper.make_node('Identity', ['input'], ['output'])],
                          'test-only', [input_info], [output_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 10
onnx.save(model, 'build/test_identity.onnx')
PY
cmake -S . -B build -DADAPTIVESERVE_TEST_ONNX="$PWD/build/test_identity.onnx"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For an end-to-end test of real pretrained inference, run the reproduction script
and then upload an actual image through `/api/predict`. Exact prediction,
confidence, selected model and latency depend on the input, training and machine;
they are deliberately not prescribed as constants in the documentation.
