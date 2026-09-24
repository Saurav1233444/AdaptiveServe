# Native inference subsystem

The native backend is a C++20 ONNX Runtime engine exposed to Python as
`adaptive_core`. It validates the model registry when constructed but loads each
ONNX file only on its first prediction. Missing model artifacts therefore do not
prevent the API from starting and displaying the registry.

Build the pinned ONNX Runtime 1.22.0 SDK and extension from the repository root:

```bash
./scripts/setup_native.sh
PYTHONPATH=build .venv/bin/python -c "import adaptive_core; print(adaptive_core.__doc__)"
```

Python owns image decoding and weight-specific preprocessing. The native call
accepts a contiguous `numpy.float32` tensor with shape `[1, 3, input_size,
input_size]`:

```python
import adaptive_core

engine = adaptive_core.Engine("configs/models.json", threads=1)
result = engine.predict("mobilenet_v3_small", tensor)
engine.unload("mobilenet_v3_small")
```

`inference_ms` measures only the synchronous `Ort::Session::Run`. `load_ms` is
non-zero for the call that creates a session and zero for a cache hit.
`memory_mb` is resident memory for the current process after inference.
`cpu_percent` is process CPU time divided by inference wall time, where one fully
busy core is 100 percent. These per-request observations are not calibrated model
profiles or research results.

The same engine is available as a JSON-lines worker. It reads one object per line
from standard input with a flat `tensor` array and writes one result object per
line to standard output. Structured operational logs go to standard error.

```bash
build/adaptive_worker --registry configs/models.json \
  --model mobilenet_v3_small --threads 1
```

The scheduler admits at most `threads` simultaneous calls and rejects excess
calls with an explicit overload error. Session handles use shared ownership, so
an unload can remove a cached session while a prediction already using that
session finishes safely.

Run native tests with:

```bash
ctest --test-dir build --output-on-failure
```

For a separate real-runtime smoke test, configure
`-DADAPTIVESERVE_TEST_ONNX=/absolute/path/to/tiny.onnx`. The fixture must accept a
float32 `[1,3,2,2]` input. This check validates ONNX Runtime execution and cache
lifecycle only; it is deliberately separate from experiment metrics.
