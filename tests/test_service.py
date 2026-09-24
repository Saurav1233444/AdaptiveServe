from PIL import Image

from adaptiveserve.service import InferenceService


def test_adaptive_path_executes_only_selected_model_and_reports_measured_violation(tmp_path):
    calls = []
    class Runtime:
        models = {"tiny": {}}
        def predict(self, name, image):
            calls.append(name)
            return {"class_index": 1, "confidence": .7, "inference_ms": 1, "selected_model": name}
    class Analyzer:
        def analyze(self, image):
            return {"score": .2, "analyzer_ms": .1}
    class Router:
        def select(self, context):
            assert context["complexity"] == .2
            assert context["cpu_available"] == .5
            return {"model": "tiny", "reason": "fixture", "constraint_satisfied": True}
    service = InferenceService(tmp_path)
    service.runtime = Runtime()
    service._adaptive = lambda policy: (Analyzer(), Router())
    result = service.predict(Image.new("RGB", (8, 8)), "learned", 50, .001, .5)
    assert calls == ["tiny"]
    assert result["complexity"] == .2
    assert result["profile_constraint_satisfied"] is True
    assert result["constraint_satisfied"] is False
    assert result["latency_ms"] > 0
