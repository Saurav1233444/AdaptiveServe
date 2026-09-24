"""Upload the existing app and trained artifacts to a free Docker Space."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def stage(destination):
    for name in ['Dockerfile', '.dockerignore', 'CMakeLists.txt', 'pyproject.toml',
                 'uv.lock', 'requirements.txt']:
        shutil.copy2(ROOT / name, destination / name)
    for name in ['adaptiveserve', 'experiments', 'core', 'workers', 'tests', 'scripts', 'configs']:
        shutil.copytree(ROOT / name, destination / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'frontend', destination / 'frontend', ignore=shutil.ignore_patterns(
        'node_modules', 'dist', '.vercel', '.env.local', '.env.*.local'))
    shutil.copytree(ROOT / 'results/demo', destination / 'results/demo')
    models = destination / 'models'
    models.mkdir()
    for name in ['mobilenet_v3_small.onnx', 'resnet50.onnx', 'efficientnet_b0.onnx',
                 'complexity_predictor.pt', 'router.pt', 'profiles.json', 'labels.json']:
        source = ROOT / 'models' / name
        if not source.is_file():
            raise FileNotFoundError(f'Required deployed artifact missing: {source}')
        shutil.copy2(source, models / name)
    shutil.copy2(ROOT / 'deploy/space.README.md', destination / 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', help='Hugging Face account/space-name')
    parser.add_argument('--dry-run', action='store_true', help='Validate deployment package without publishing')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='adaptiveserve-deploy-') as directory:
        folder = Path(directory)
        stage(folder)
        size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        print(f'Validated deployment package: {size / 1024**2:.1f} MiB')
        if args.dry_run:
            return
        if not args.repo or '/' not in args.repo:
            parser.error('--repo account/space-name is required')
        from huggingface_hub import HfApi
        api = HfApi()  # Reads HF_TOKEN or the local authenticated token store.
        api.whoami()  # Fail before creating anything if login is missing.
        api.create_repo(args.repo, repo_type='space', space_sdk='docker', exist_ok=True,
                        private=False, space_hardware='cpu-basic')
        api.add_space_variable(args.repo, 'ADAPTIVESERVE_CORS_ORIGINS', '*')
        api.upload_folder(repo_id=args.repo, repo_type='space', folder_path=folder,
                          commit_message='Deploy existing AdaptiveServe native research prototype')
        info = api.space_info(args.repo)
        result = {'space': f'https://huggingface.co/spaces/{args.repo}',
                  'backend': getattr(info, 'host', None), 'status': 'uploaded; build pending'}
        (ROOT / 'deployment-status.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
