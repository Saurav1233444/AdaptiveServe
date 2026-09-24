# AdaptiveServe research design

Implement the user's supplied architecture in this checkout. The existing worker
responses and registry numbers are placeholders and must be replaced, not reused
as observations. No benchmark data is generated until real inference runs.

## Method

ImageNet pretrained MobileNetV3-Small, ResNet50 and EfficientNet-B0 retain their
1,000-class heads. Use Imagenette (an ImageNet subset with explicit WordNet to
ImageNet index mapping) or a manifest of locally supplied ImageNet images. Score
the complete 1,000-class output, not a reduced candidate set. Use each weight's
documented preprocessing. Training, calibration and test image IDs are disjoint.

A small CNN plus six image statistics predicts difficulty, supervised by the
mean true-class probability deficit of all three models on training images.
A small neural router estimates model utilities from difficulty, latency budget,
CPU availability and memory budget. Training targets use observed per-image
correctness and latency; resource contexts are explicitly simulated budgets.
Infeasible model profiles are masked; fallback violations are visible. Learned
and threshold routers are compared against all three static models. Timing
includes analysis and routing overhead for adaptive policies.

## Interfaces

- Python package: `adaptiveserve/`; FastAPI: `adaptiveserve.api:app`.
- Registry: `configs/models.json`, `models` array, ordered names
  `mobilenet_v3_small`, `resnet50`, `efficientnet_b0`. Fields: `name`,
  `display_name`, `onnx_path`, `input_size` (224), `resize_size` (256),
  `interpolation` (`bilinear` or `bicubic`), `accuracy`, `latency_ms`, `memory_mb`,
  `input_type`, `endpoint`, `weights`. Measurements are null until calibrated.
  ONNX paths resolve relative to the repository root (registry parent parent).
- Native module `adaptive_core`, built to `build/`, exposes
  `Engine(registry_path: str, threads: int=1)`,
  `predict(model_name: str, tensor: numpy.float32[1,3,224,224]) -> dict`
  containing `logits` (flat list), `inference_ms`, `load_ms`,
  `memory_mb` (process RSS), `cpu_percent` (process CPU/wall, one core=100%).
  `unload(name)` releases a session; `loaded_models()` lists cached sessions.
  Native calls release the GIL; bounded execution rejects overload explicitly.
- Python `ModelRuntime(config_path, threads=1, backend='cpp')` handles
  preprocessing and returns prediction/class index/confidence plus native timing.
  Optional ONNX Python backend is labeled separately; never silently substitutes.
- Analyzer `ComplexityAnalyzer(checkpoint)` with `analyze(PIL.Image)` returns
  `score`, `features`, `embedding`, `analyzer_ms`, `trained`. Missing trained
  checkpoints raise actionable errors for learned inference.
- Router `RuleRouter(profiles).select(context)` and
  `LearnedRouter(checkpoint, profiles).select(context)` return `model`, `reason`,
  `constraint_satisfied`; context has `complexity`, `latency_budget_ms`,
  `cpu_available` (0..1), `memory_budget_mb`.
- Dataset manifest CSV: `sample_id,path,label,split`; path relative to manifest,
  ImageNet integer label 0..999; split train/calibration/test.
- Profile observations CSV: `sample_id,model,label,prediction,confidence,
  true_probability,inference_ms,latency_ms,memory_mb,cpu_percent`.
- API: GET `/api/health`, `/api/models`, `/api/metrics`, `/api/results`,
  POST `/api/predict` multipart `file`, `policy` (`learned`,`rule`,or model name),
  `latency_budget_ms`, `memory_budget_mb`. Response includes `request_id`,
  `complexity`, `selected_model`, `reason`, `prediction`, `class_index`,
  `confidence`, `latency_ms`, `inference_ms`, `analyzer_ms`, `cpu_percent`,
  `memory_mb`, `constraint_satisfied`, `backend`.
- Results API: `{available, summary: [...], plots: [{name,url}], metadata}`.
  Summary fields `policy,n,accuracy,latency_ms,p95_latency_ms,throughput_rps,
  memory_mb,cpu_percent,sla_violation_rate`. Empty states contain no invented data.
- Metrics API: `{requests,average_latency_ms,current_model,cpu_percent,memory_mb,
  recent: [...],model_counts: {...}}`.

## Validation and interpretation

Unit tests cover each module, malformed inputs, missing artifacts, resource
constraints, dataset leakage and metric aggregation. Integration runs export real
weights, validate ONNX logits against PyTorch, train artifacts, benchmark held-out
images, generate plots, build the UI, and exercise an uploaded image through API.
Small runs demonstrate reproducibility and are not evidence of general superiority.
Report negative improvements and uncertainty honestly; document resident-process
memory, batch-one sequential throughput, warmup, thread counts and dataset limits.
