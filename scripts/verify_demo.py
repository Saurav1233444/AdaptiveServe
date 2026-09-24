"""Verify real pretrained inference through the API; requires prepared artifacts."""
import json

import pandas as pd
from fastapi.testclient import TestClient

from adaptiveserve.api import create_app
from adaptiveserve.runtime import ROOT


def main():
    client = TestClient(create_app())
    assert client.get('/api/health').json()['ready']
    row = pd.read_csv(ROOT / 'data/manifest.csv').query("split == 'test'").iloc[0]
    image = ROOT / 'data' / row.path
    report = []
    for policy in ['mobilenet_v3_small', 'resnet50', 'efficientnet_b0', 'rule', 'learned']:
        response = client.post('/api/predict', data={'policy': policy},
                               files={'file': (image.name, image.read_bytes(), 'image/jpeg')})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['backend'] == 'cpp' and 0 <= result['confidence'] <= 1
        assert result['inference_ms'] > 0
        if policy in ['rule', 'learned']:
            assert 0 <= result['complexity'] <= 1
        report.append({key: result[key] for key in [
            'policy', 'selected_model', 'prediction', 'confidence', 'complexity',
            'inference_ms', 'latency_ms', 'backend']})
    for path in ['/', '/models', '/monitoring', '/results', '/architecture']:
        assert client.get(path).status_code == 200, path
    results = client.get('/api/results').json()
    assert results['available'] and len(results['summary']) == 5
    assert len(results['plots']) == 8
    for plot in results['plots']:
        assert client.get(plot['url']).status_code == 200
    destination = ROOT / 'docs/smoke-results.json'
    destination.write_text(json.dumps({'sample_id': str(row.sample_id),
                                      'image': str(row.path), 'predictions': report}, indent=2) + '\n')
    print(f'PASS: five real C++ policies, five dashboard routes, eight figures; {destination}')


if __name__ == '__main__':
    main()
