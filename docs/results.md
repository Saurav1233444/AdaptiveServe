# Measured demonstration results

Completed on 14 September 2026 (Asia/Kolkata). These are observations from
this machine, not promised performance on another host.

Training used 300 Imagenette images, calibration used 100 separate training
images, and evaluation used 100 validation images with three repetitions per
policy: 1,500 requests. The seed was 42, native ONNX Runtime used one thread,
and routing context was 50 ms / 2,048 MiB / CPU availability 1.0. Images retain
1,000-class ImageNet heads. Adaptive request timing includes the analyzer.

| Policy | Accuracy | Mean latency (ms) | Throughput (req/s) | RSS (MiB) | CPU (%) |
|---|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 67% | 9.20 | 76.68 | 131.4 | 99.5 |
| resnet50 | 79% | 156.75 | 6.14 | 292.1 | 99.4 |
| efficientnet_b0 | 79% | 58.71 | 14.54 | 171.5 | 98.5 |
| rule | 74% | 26.12 | 32.25 | 478.7 | 98.9 |
| learned | 67% | 15.99 | 45.84 | 481.5 | 98.4 |

The threshold router provides an intermediate operating point: 74% accuracy
at 26.12 ms, versus MobileNet's 67% at 9.20 ms and EfficientNet's 79% at
58.71 ms. This is an observed tradeoff, not a universal improvement.

The learned router selected MobileNet for all 300 requests under this context.
It matched MobileNet accuracy but added 73.8% latency. Relative to ResNet it
reduced latency by 89.8% while losing 12 accuracy percentage points. This is a
negative result for the learned policy relative to the cheapest static model;
more training epochs alone do not establish an improved selection policy.

All policies consumed roughly one busy CPU core. Adaptive RSS includes cached
candidate sessions and PyTorch, so it substantially exceeds static process RSS.
Sequential wall throughput includes image decode and SQLite feedback. Budget
inputs do not throttle hardware. Laptop load and thermal state affect timing;
these runs were not made on a dedicated isolated benchmark host.

Accuracy confidence intervals count 100 unique images, not 300 repeated requests.
The intervals and p95 latency are in the saved summary. The small test set and
single seed are sufficient for a working demonstration, not general superiority.

## Retained evidence

- [Summary and machine/source provenance](../results/demo/summary.json)
- [All 1,500 per-request observations](../results/demo/observations.csv)
- [Latency CSV](../results/demo/latency.csv)
- [Accuracy CSV](../results/demo/accuracy.csv)
- [Memory CSV](../results/demo/memory.csv)
- [Model selections](../results/demo/model_selection.csv)
- [Actual five-policy API smoke responses](smoke-results.json)

Regenerate all eight figures from the retained observations:

```bash
.venv/bin/python -m experiments.plot --results results/demo --output frontend/public/results
```

The CPU figure was added after evaluation and uses the same saved CPU
measurements. Benchmark provenance preserves the source hashes at execution.
Smoke-test timing includes first-load costs and is separate from the warmed
benchmark. No unit-test fixture values appear in these results.
