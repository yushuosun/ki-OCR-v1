"""Fixed public-model preload verification/copy. Standard library only.

No HF import, model_info, token lookup, network call or inference. Extra source
files are ignored; only the exact frozen file inventory may be copied.
"""
import re
import shutil
from pathlib import Path, PurePosixPath
from assets import ROOT, MODELS, load, model_pins, sha

SCHEMA = 'KI_OCR_PUBLIC_MODELS_OFFLINE_V1'

def canonical_manifest(document):
    if not isinstance(document,dict) or document.get('schema') != SCHEMA or not isinstance(document.get('models'), list):
        raise ValueError('Unsupported fixed public model manifest')
    seen, inventory = set(), {}
    for model in document['models']:
        if not isinstance(model,dict):
            raise ValueError('Public model manifest entries must be objects')
        kind = model.get('kind')
        if not isinstance(kind,str) or kind not in MODELS or kind in seen:
            raise ValueError('Unknown/duplicate public model kind')
        seen.add(kind)
        repository, revision = MODELS[kind]
        if (model.get('repository'), model.get('revision')) != (repository, revision) or model.get('public_ungated') is not True:
            raise ValueError('Fixed public repository/revision mismatch: ' + kind)
        pins = model_pins(kind)
        files = model.get('files')
        if not isinstance(files, list):
            raise ValueError('Public model file inventory required')
        names = set()
        for item in files:
            if not isinstance(item,dict):
                raise ValueError('Public model file entries must be objects')
            name = item.get('file')
            if not isinstance(name, str) or not name or '\\' in name or ':' in name:
                raise ValueError('Unsafe public model filename')
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or path.as_posix() != name or name in names or name not in pins:
                raise ValueError('Unlisted/duplicate/unsafe public model filename')
            names.add(name)
            pin = pins[name]
            expected_sha = pin['sha256'] if isinstance(pin, dict) else pin
            size = item.get('bytes')
            if type(size) is not int or size < 0 or item.get('sha256') != expected_sha or not re.fullmatch('[0-9a-f]{64}', expected_sha):
                raise ValueError('Invalid fixed size/hash: ' + kind + '/' + name)
            url = 'https://huggingface.co/' + repository + '/resolve/' + revision + '/' + name
            if item.get('url') != url:
                raise ValueError('Fixed public URL mismatch')
            inventory[kind + '/' + name] = {'kind': kind, 'repository': repository,
                'revision': revision, 'file': name, 'bytes': size, 'sha256': expected_sha, 'url': url}
        if names != set(pins):
            raise ValueError('Missing frozen model manifest files: ' + kind)
    if seen != set(MODELS):
        raise ValueError('All three fixed public model groups required')
    return inventory

def fixed_inventory(manifest=None):
    frozen = canonical_manifest(load(ROOT / 'OFFLINE_MODEL_MANIFEST.json'))
    if manifest is not None and canonical_manifest(load(manifest)) != frozen:
        raise ValueError('Preloaded manifest size/revision/hash differs from frozen public manifest')
    return frozen

def safe_file(directory, name):
    directory = Path(directory).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Explicit regular public model directory required')
    path = directory / name
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Missing/outside preloaded public asset: ' + name)
    return path

def verify_preloaded_models(directory, manifest=None):
    inventory = fixed_inventory(manifest)
    directory = Path(directory).absolute()
    for name, expected in inventory.items():
        path = safe_file(directory, name)
        if path.stat().st_size != expected['bytes']:
            raise ValueError('Preloaded public asset size mismatch: ' + name)
        if sha(path) != expected['sha256']:
            raise ValueError('Preloaded public asset SHA mismatch: ' + name)
    return {'status': 'OFFLINE_PUBLIC_MODEL_FILES_VERIFIED_NOT_INFERENCE',
            'directory': str(directory), 'files_verified': len(inventory),
            'bytes_verified': sum(p['bytes'] for p in inventory.values()),
            'repositories': [{'kind': k, 'repository': r, 'revision': v} for k,(r,v) in MODELS.items()],
            'manifest_sha256': sha(ROOT/'OFFLINE_MODEL_MANIFEST.json'),
            'network_calls': 0, 'token_lookup': False, 'GPU_started': False,
            'provenance': 'Previously downloaded fixed public assets, verified locally; not a new network download'}

def import_preloaded_models(directory, root, manifest=None):
    verified = verify_preloaded_models(directory, manifest)
    inventory = fixed_inventory(manifest)
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError('Explicit regular installation root required')
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'models'
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Outside/symlink destination model directory')
        verify_preloaded_models(target, manifest)
        return dict(verified, status='OFFLINE_PUBLIC_MODELS_REUSED_VERIFIED', target=str(target), copied_files=0)
    # Verify every source first. Exclusive target creation prevents overwriting
    # another install. Late I/O failure retains partial files and fails; they are
    # never resumed/merged or described as a completed import.
    target.mkdir(exist_ok=False)
    for name in inventory:
        src = safe_file(directory, name)
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError('Destination escaped installation root')
        with src.open('rb') as source, dst.open('xb') as output:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    verify_preloaded_models(target, manifest)
    return dict(verified, status='OFFLINE_PUBLIC_MODELS_IMPORTED_VERIFIED',
                target=str(target), copied_files=len(inventory))
