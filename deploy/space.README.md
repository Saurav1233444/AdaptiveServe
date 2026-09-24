---
title: AdaptiveServe
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
suggested_hardware: cpu-basic
---

# AdaptiveServe public research demo

Real C++ ONNX inference with MobileNetV3, ResNet50 and EfficientNet-B0,
a trained difficulty predictor and a learned model router.

API: `/health`, `/api/health`, `/api/models`, `/api/predict`, `/api/metrics`,
`/api/results`. Interactive API documentation: `/docs`.

Upload an image in the dashboard to start. Free CPU hosting may sleep and take
some time to wake. Request metrics reset on restart. Results shown are retained
local research measurements, not a cloud performance claim.

Pretrained weights were prepared with torchvision ImageNet V1 weights. The
original model terms and dataset terms apply; research code is accompanied by
its source. Uploaded images are processed in memory and are not retained.
