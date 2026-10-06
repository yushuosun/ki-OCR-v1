"""CPU-only public asset identities. Never access tokens or private caches."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    'tele': ('XingChen-AGI/TeleOCR', 'e92585356c0d0b7b7a65938f3da035c6593cc9a6'),
    'paddle': ('PaddlePaddle/PaddleOCR-VL-1.6', 'c5630abae1d940eafe0697512a0325494b02ab42'),
    'paddle_layout': ('PaddlePaddle/PP-DocLayoutV3', '7b48a7566925fa464281f930c58eee04fe2c862a'),
}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(directory, pins):
    directory = Path(directory).resolve()
    for name, pin in pins.items():
        path = directory / name
        expected = pin['sha256'] if isinstance(pin, dict) else pin
        if not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError('Missing/outside pinned asset: ' + name)
        if sha(path) != expected:
            raise ValueError('Pinned asset SHA mismatch: ' + name)
    return len(pins)


def model_pins(kind):
    if kind == 'tele':
        return load(ROOT / 'MODEL_PINS.json')['tele']
    prefix = 'models/' + kind + '/'
    return {name.removeprefix(prefix): value for name, value in
            load(ROOT / 'PADDLE_MODEL_PINS.json')['pins'].items() if name.startswith(prefix)}


def clean_env(root, native_lib=None):
    env = dict(os.environ)
    # Do not inherit credentials or alternate registries into public downloads.
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_CACHE',
                'HUGGINGFACE_HUB_CACHE', 'HF_ENDPOINT', 'PYTHONPATH', 'PYTHONHOME',
                'VIRTUAL_ENV', 'CONDA_PREFIX','VLLM_PLUGINS','VLLM_RPC_BASE_PATH'):
        env.pop(key, None)
    env.update(HF_HOME=str(Path(root) / 'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               PADDLE_PDX_CACHE_HOME=str(Path(root) / 'paddlex'),
               VLLM_CACHE_ROOT=str(Path(root) / 'vllm'),
               VLLM_ASSETS_CACHE=str(Path(root) / 'vllm/assets'),
               VLLM_CONFIG_ROOT=str(Path(root) / 'vllm-config'),
               TORCH_HOME=str(Path(root) / 'torch'),
               XDG_CACHE_HOME=str(Path(root) / 'xdg'),
               HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONUNBUFFERED='1')
    if native_lib is not None:
        env.pop('LD_PRELOAD', None)
        env['LD_LIBRARY_PATH'] = str(native_lib)
    return env


def download_models(root):
    # Imported only inside the newly installed CPU venv.
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root).resolve()
    report = []
    for kind, (repository, revision) in MODELS.items():
        info = HfApi(token=False).model_info(repository, revision=revision, token=False)
        if info.private or info.gated:
            raise ValueError('Public ungated model required; no agreement accepted: ' + repository)
        if info.sha != revision:
            raise ValueError('Public revision mismatch: ' + repository)
        target = root / 'models' / kind
        snapshot_download(repository, revision=revision, token=False,
                          local_dir=str(target), cache_dir=str(root / 'download-cache'),
                          allow_patterns=list(model_pins(kind)))
        count = verify_pins(target, model_pins(kind))
        report.append({'repository': repository, 'revision': revision, 'files_verified': count,
                       'weights_downloaded': True, 'agreements_accepted': False})
        write(root / 'MODEL_DOWNLOADS.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--download', type=Path)
    action.add_argument('--verify-preloaded-models', type=Path)
    action.add_argument('--import-preloaded-models', type=Path)
    parser.add_argument('--root', type=Path, help='Installation root for local model import')
    parser.add_argument('--model-manifest', type=Path, help='Optional supplied manifest; must equal the frozen public inventory')
    parser.add_argument('--output', type=Path, help='Optional new local verification receipt')
    args = parser.parse_args()
    if args.download:
        sanitized = clean_env(args.download)
        os.environ.clear()
        os.environ.update(sanitized)
        result = download_models(args.download)
    else:
        from offline_assets import verify_preloaded_models, import_preloaded_models
        if args.import_preloaded_models:
            if args.root is None:
                parser.error('--import-preloaded-models requires --root')
            result = import_preloaded_models(args.import_preloaded_models, args.root, args.model_manifest)
        else:
            result = verify_preloaded_models(args.verify_preloaded_models, args.model_manifest)
    if args.output:
        if args.output.exists():
            raise ValueError('New verification receipt path required')
        write(args.output, result)
    print(json.dumps(result, indent=2))
