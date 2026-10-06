"""Validate complete installed environments and provision public models offline.

No pip install, native download, Git command, HF request or admission operation.
Incomplete environments stop; this is verified continuation, not a clean install.
"""
import os
import subprocess
import time
import uuid
from pathlib import Path
from assets import ROOT, clean_env, load, verify_pins, write
from offline_assets import verify_preloaded_models, import_preloaded_models

def reuse_with_preloaded(root, preloaded, manifest=None, native_prefix=None):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Complete existing installation root required for offline continuation')
    source_check = verify_preloaded_models(preloaded, manifest)
    runtime_path = root / 'runtime.json'
    old_runtime = load(runtime_path) if runtime_path.is_file() else None
    prefix = Path(native_prefix or (old_runtime or {}).get('native_prefix') or root/'native').absolute()
    verify_pins(prefix, load(ROOT/'NATIVE_RUNTIME_LOCK.json')['library_pins'])
    deps = load(ROOT/'DEPENDENCIES.json')
    tele_source = root / 'TeleOCR'
    verify_pins(tele_source, deps['tele_source_pins'])
    verify_pins(tele_source, load(ROOT/'UPSTREAM_PLUGIN_PINS.json'))
    from inference import verify_environment
    environments = {}
    for kind in ('cpu', 'tele', 'paddle'):
        python = root / (kind+'_env') / 'bin/python'
        if not python.is_file():
            raise ValueError('Complete existing isolated environment required: '+kind)
        environments[kind] = verify_environment(python, kind)
    policy = (old_runtime or {}).get('execution_policy', 'standalone')
    if policy not in {'standalone', 'cluster'}:
        raise ValueError('Existing runtime execution policy is invalid')
    runtime = {'tele_source': str(tele_source), 'tele_model': str(root/'models/tele'),
        'paddle_root': str(root), 'tele_python': str(root/'tele_env/bin/python'),
        'paddle_python': str(root/'paddle_env/bin/python'), 'native_prefix': str(prefix),
        'installation_root': str(root), 'execution_policy': policy}
    if old_runtime:
        for key, value in runtime.items():
            if old_runtime.get(key) != value:
                raise ValueError('Existing runtime conflicts with explicit verified installation: '+key)
    run = root / 'installer-runs' / ('offline-'+uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    imports = []
    env = clean_env(run, prefix/'lib')
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    try:
        for kind in ('tele', 'paddle'):
            log = run/(kind+'_import.log')
            print('OFFLINE_CPU_IMPORT_START '+kind, flush=True)
            before = time.monotonic()
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.run([str(root/(kind+'_env')/'bin/python'), '-I',
                    str(ROOT/'import_preflight.py'), '--kind', kind, '--native-prefix', str(prefix),
                    '--tele-source', str(tele_source), '--output', str(run/(kind+'_import.json')), '--strict-offline'],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=300)
            imports.append({'kind':kind, 'exit':child.returncode,
                            'seconds':time.monotonic()-before, 'log':str(log)})
            if child.returncode:
                raise RuntimeError('Existing environment CPU import failed: '+kind+'; see '+str(log))
            imported=load(run/(kind+'_import.json'))
            if imported.get('status') != 'CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE' or imported.get('strict_offline') is not True or imported.get('offline_guard',{}).get('attempts') != {'network':0,'model_info_or_download':0,'token_lookup':0}:
                raise ValueError('Actual CPU import preflight did not pass: '+kind)
        model_result = import_preloaded_models(preloaded, root, manifest)
        if old_runtime is None:
            if runtime_path.exists():
                raise ValueError('Runtime appeared during continuation; refusing overwrite')
            write(runtime_path, runtime)
        result = {'status':'INSTALLED_ENVIRONMENTS_REUSED_PUBLIC_MODELS_VERIFIED',
            'runtime':str(runtime_path), 'verification_run':str(run), 'environments':environments,
            'CPU_imports':imports, 'model_result':model_result,
            'source_check':source_check, 'seconds':time.monotonic()-started,
            'environment_reused':True, 'fresh_uninterrupted_install':False,
            'network_calls':0, 'token_lookup':False, 'GPU_started':False,
            'admission_or_registration':False, 'full_inference_acceptance':False}
        write(run/'OFFLINE_INSTALL_RESULT.json', result)
        return result
    except Exception as error:
        write(run/'OFFLINE_INSTALL_FAILURE.json', {'status':'OFFLINE_CONTINUATION_FAILED',
              'error_type':type(error).__name__, 'error':str(error), 'CPU_imports':imports,
              'GPU_started':False, 'automatic_retry':False})
        raise
