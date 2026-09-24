import io

from fastapi.testclient import TestClient
from PIL import Image

from adaptiveserve.api import create_app


class StubService:
    """Only test fixtures use a stand-in; the application has no mock backend."""
    backend = "test-fixture"
    def health(self):
        return {"status": "ok", "ready": True, "backend": self.backend, "detail": "fixture"}
    def models(self):
        return []
    def predict(self, image, policy, latency_budget_ms, memory_budget_mb):
        assert image.mode == "RGB"
        return {"request_id": "test", "selected_model": "fixture", "latency_ms": 1}


def test_api_upload_validation_and_feedback(tmp_path):
    client = TestClient(create_app(root=tmp_path, service=StubService()))
    assert client.get("/api/health").json()["ready"]
    assert client.get("/api/results").json()["available"] is False
    assert client.post("/api/predict", files={"file": ("bad.png", b"invalid")}).status_code == 400
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, "PNG")
    response = client.post("/api/predict", files={"file": ("input.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert client.get("/api/metrics").json()["requests"] == 1
    assert client.post("/api/predict", data={"latency_budget_ms": -1},
                       files={"file": ("input.png", image.getvalue())}).status_code == 422


def test_missing_models_explain_setup(tmp_path):
    client = TestClient(create_app(root=tmp_path))
    assert client.get("/api/health").json()["ready"] is False


def test_deployment_health_and_cors(tmp_path, monkeypatch):
    monkeypatch.setenv('ADAPTIVESERVE_CORS_ORIGINS', 'https://demo.vercel.app')
    client = TestClient(create_app(root=tmp_path, service=StubService()))
    assert client.get('/health').json()['ready'] is True
    response = client.options('/api/predict', headers={
        'Origin': 'https://demo.vercel.app',
        'Access-Control-Request-Method': 'POST',
    })
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'https://demo.vercel.app'
    assert client.options('/api/predict', headers={
        'Origin': 'https://unrelated.example',
        'Access-Control-Request-Method': 'POST',
    }).status_code == 400


def test_dashboard_routes_support_direct_navigation_and_keep_plot_assets(tmp_path):
    dist = tmp_path / "frontend/dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>Research dashboard</html>")
    plots = tmp_path / "frontend/public/results"
    plots.mkdir(parents=True)
    (plots / "latency.png").write_bytes(b"plot-fixture")
    client = TestClient(create_app(root=tmp_path, service=StubService()))
    for route in ["/", "/models", "/monitoring", "/results", "/architecture"]:
        response = client.get(route)
        assert response.status_code == 200, route
        assert "Research dashboard" in response.text
    assert client.get("/results/latency.png").content == b"plot-fixture"
    assert client.get("/api/unknown").status_code == 404
