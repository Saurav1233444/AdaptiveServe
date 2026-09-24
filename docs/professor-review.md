# Three-minute professor review

**0:00–0:35 — Question.** “Large models are not necessary for every image, but
a cheap model can lose accuracy. I am studying whether input difficulty and
resource context can select a better accuracy–latency operating point.” Show
the Architecture page and point to the analyzer, router and three teachers.

**0:35–1:15 — Demonstration.** Upload a test image. Explain the learned difficulty
score, selected model and reason. Point out the prediction and actual end-to-end
latency. Change the latency budget and repeat; model selection can change, or
the system can explicitly report that no measured profile fits. A single image
is illustrative; it does not establish performance.

**1:15–2:10 — Method and evidence.** Show Research Results. All three static models
and two adaptive policies used the same held-out images. Explain train/calibration/
test separation, the full ImageNet output head, and that adaptive timing includes
the CNN and router. Compare accuracy and latency together, then show memory.
Read the actual values and confidence interval; include negative results.

**2:10–2:40 — Contribution.** “The prototype connects a CNN difficulty predictor
and learned resource-aware router to real C++ ONNX inference. It provides
reproducible per-request observations, static baselines, and visible routing
overhead rather than fixed demonstration metrics.”

**2:40–3:00 — Limits and next experiment.** A 100-image test subset supports a
pipeline demonstration, not a general performance claim. Cached candidates
increase memory. CPU budget contexts do not emulate actual throttling. Next use
larger datasets, more seeds, measured contention and confidence-cascade/oracle
baselines. Do not describe the selected model as globally optimal.
