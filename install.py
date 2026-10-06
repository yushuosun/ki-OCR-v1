"""One-command, venv-only Linux installation. No Docker, sudo or GPU calls."""
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path
from assets import ROOT, clean_env, load, sha, verify_pins, write

PADDLE_INDEX = 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'


def python_at(env):
    # Keep lexical venv executable: resolving its symlink would select base Python.
    return Path(env).absolute() / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')


def install(root, components, models, pip_cache=None, preloaded_models=None,
            model_manifest=None, reuse_environments=False, native_prefix=None,
            resume_install=False, resume_receipt_sha256=None, resume_timeout_seconds=10800):
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise ValueError('Inference installation requires Linux x86_64 (CPU tests work on Windows).')
    if sys.version_info[:2] != (3, 10):
        raise ValueError('Use Python 3.10; the verified frozen Paddle wheel targets cp310.')
    if resume_install and (reuse_environments or native_prefix is not None
                          or list(components) != ['cpu','tele','paddle'] or not resume_receipt_sha256):
        raise ValueError('Resume requires all components, original receipt SHA and no reuse/native override')
    if resume_receipt_sha256 and not resume_install:
        raise ValueError('Receipt SHA requires explicit --resume-install')
    if preloaded_models is not None:
        if models or list(components) != ['cpu','tele','paddle']:
            raise ValueError('Preloaded models require all components and no download option')
        if reuse_environments:
            if pip_cache is not None:
                raise ValueError('Offline environment reuse requires no download/cache option')
            from offline_install import reuse_with_preloaded
            return reuse_with_preloaded(root, preloaded_models, model_manifest, native_prefix)
        if native_prefix is not None:
            raise ValueError('Fresh install must create its own native runtime; --native-prefix requires reuse')
        root = Path(root).absolute()
        if not resume_install and (root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir())))):
            raise ValueError('Fresh preloaded installation requires a new or completely empty root')
        from offline_assets import verify_preloaded_models
        # Check the fixed public inventory before network/install side effects.
        verify_preloaded_models(preloaded_models, model_manifest)
    elif reuse_environments or model_manifest is not None or native_prefix is not None:
        raise ValueError('Reuse/manifest/native-prefix options require explicit offline --preloaded-models')
    root = root.absolute()
    pip_cache = Path(pip_cache).absolute() if pip_cache else root / 'pip-cache'
    resume = None
    if resume_install:
        from resume_install import CloneResume
        resume = CloneResume(root, resume_receipt_sha256, pip_cache, resume_timeout_seconds)
    tele_source = resume.source if resume else root/'TeleOCR'
    root.mkdir(parents=True, exist_ok=True)
    from native_setup import setup
    if not (root/'native').exists():
        setup(root)
    verify_pins(root/'native',load(ROOT/'NATIVE_RUNTIME_LOCK.json')['library_pins'])
    env = clean_env(root, root/'native/lib')
    env['CUDA_VISIBLE_DEVICES'] = ''
    if preloaded_models is not None:
        # Dependency downloads remain online; only model provisioning/imports
        # avoid HF and PaddleX's import-time hoster connectivity probe.
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    records = []
    installer_pid = os.getpid()
    result_path = resume.result_path if resume else root/'INSTALL_RESULT.json'

    def save(value):
        if resume:
            for key, fact in resume.facts().items():
                value.setdefault(key,fact)
        write(result_path, value)

    def command(argv, name):
        argv = [str(x) for x in argv]
        if resume:
            old = resume.skipped(name, argv)
            if old is not None:
                records.append(old)
                save({'commands':records,'GPU_started':False,'status':'RESUME_RUNNING'})
                print(json.dumps(old), flush=True)
                return
        started = time.monotonic()
        logfile = (resume.run if resume else root) / (name + '.log')
        if logfile.exists():
            raise ValueError('Existing installation attempt: ' + str(logfile) + '; choose a new root.')
        with logfile.open('x', encoding='utf-8') as stream:
            if resume and time.monotonic() >= resume.deadline:
                raise TimeoutError('Explicit resume wall bound reached; no automatic retry')
            process = subprocess.Popen(argv, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       **({'start_new_session':True} if resume else {}))
            save({'commands': records, 'GPU_started': False,
                  'installer_pid': installer_pid, 'active_pid': process.pid,
                  'active_argv': argv, 'active_log': str(logfile.relative_to(root))})
            timed_out = False
            if resume:
                try:
                    process.wait(timeout=max(0.001,resume.deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL);process.wait()
            else:
                process.wait()
        record = {'argv': argv, 'exit': process.returncode,
                  'seconds': time.monotonic() - started, 'log': str(logfile.relative_to(root)),
                  'log_sha256': sha(logfile)}
        if resume:
            record.update(stage=name,resumed_skip=False,timeout_reached=timed_out)
        records.append(record)
        save({'commands': records, 'GPU_started': False,
              **({'status':'RESUME_RUNNING' if process.returncode==0 else 'RESUME_FAILED'} if resume else {})})
        print(json.dumps(record), flush=True)
        if timed_out:
            raise TimeoutError('Explicit resume wall bound reached: '+name)
        if process.returncode:
            raise RuntimeError('Installation failed: ' + name + '; see ' + str(logfile))

    def create(kind):
        target = root / (kind + '_env')
        if target.exists():
            if resume and kind in resume.environments:
                return python_at(target)
            raise ValueError('Fresh venv required: ' + str(target))
        # Portable CPython needs its base libpython layout; Linux venv's normal
        # executable symlink preserves that layout without inheriting packages.
        venv.EnvBuilder(with_pip=True, system_site_packages=False, symlinks=True).create(target)
        return python_at(target)

    def pip(python, arguments, name):
        command([python, '-m', 'pip', '--isolated', '--timeout', '300', '--retries', '0', *arguments], name)

    for kind in components:
        if resume:resume.require_time()
        python = create(kind)
        if kind == 'paddle':
            wheel = load(ROOT / 'PADDLE_WHEEL_PIN.json')
            pip(python, ['install', '--no-deps', '--index-url', PADDLE_INDEX,
                         '--cache-dir', pip_cache, wheel['url']+'#sha256='+wheel['sha256']],
                'paddle_cuda_install')
        pip(python, ['install', '--index-url', 'https://pypi.org/simple', '--cache-dir', pip_cache,
                     '-r', ROOT / ('requirements-' + kind + '.lock')], kind + '_install')
        if kind == 'tele':
            source = tele_source
            deps = load(ROOT / 'DEPENDENCIES.json')
            command(['git', *(['-c','http.version=HTTP/1.1'] if resume else []),
                     'clone', '--no-checkout', deps['tele_source_url'], source], 'tele_source_clone')
            command(['git', '-C', source, '-c', 'core.autocrlf=false', 'checkout', '--detach',
                     deps['tele_source_commit']], 'tele_source_checkout')
            verify_pins(source, deps['tele_source_pins'])
            verify_pins(source, load(ROOT / 'UPSTREAM_PLUGIN_PINS.json'))
            if resume:resume.require_time()
            pip(python, ['install', '--index-url', 'https://pypi.org/simple', '--no-deps', source],
                'tele_plugin_install')
        pip(python, ['check'], kind + '_pip_check')
        command([python, '-I', '-c',
                 "import json,sys,importlib.metadata as m; print(json.dumps({'prefix':sys.prefix,"
                 "'base_prefix':sys.base_prefix,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"],
                kind + '_environment')
        if kind in ('tele','paddle'):
            command([python,'-I',ROOT/'import_preflight.py','--kind',kind,
                     '--native-prefix',root/'native','--tele-source',tele_source,
                     '--output',root/(kind+'_import_preflight.json'),
                     *(['--strict-offline'] if preloaded_models is not None else [])],kind+'_import_preflight')
            if preloaded_models is not None:
                imported = load(root/(kind+'_import_preflight.json'))
                if (imported.get('status') != 'CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE'
                        or imported.get('strict_offline') is not True
                        or imported.get('offline_guard',{}).get('attempts') !=
                        {'network':0,'model_info_or_download':0,'token_lookup':0}):
                    raise ValueError('Fresh environment offline CPU import did not pass: '+kind)
    if models:
        cpu = python_at(root / 'cpu_env')
        if not cpu.is_file():
            raise ValueError('--download-models requires an installed CPU environment.')
        command([cpu, ROOT / 'assets.py', '--download', root], 'public_model_download')
    model_result = None
    if preloaded_models is not None:
        if resume:resume.require_time()
        from offline_assets import import_preloaded_models
        model_result = import_preloaded_models(preloaded_models, root, model_manifest)
        if resume:resume.require_time()
        write(root/'PRELOADED_MODEL_IMPORT.json', model_result)
    runtime = {'tele_source': str(tele_source), 'tele_model': str(root / 'models/tele'),
               'paddle_root': str(root), 'tele_python': str(python_at(root / 'tele_env')),
               'paddle_python': str(python_at(root / 'paddle_env')),
               'native_prefix': str(root/'native'),
               'installation_root': str(root), 'execution_policy': 'standalone'}
    if resume and (root/'runtime.json').exists():
        raise ValueError('Runtime appeared during continuation; refusing overwrite')
    write(root / 'runtime.json', runtime)
    result = {'status': 'INSTALL_COMMANDS_COMPLETE', 'components': components,
              'GPU_started': False, 'model_download_requested': models,
              'runtime': str(root / 'runtime.json'), 'commands': records,
              'full_inference_acceptance': False}
    if preloaded_models is not None:
        result.update(model_provisioning='fixed_public_preload', model_result=model_result,
                      environment_reused=False, fresh_root_install_commands_complete=True,
                      dependency_downloads_online=True, model_HF_requests=False,
                      admission_or_registration=False)
    if resume:
        # Reuse only this same initial root's validated CPU/Tele dependency envs.
        result.update(environment_reused=True,external_environment_reused=False,
                      prior_root_dependency_envs_reused=True,
                      HTTP1_clone_success=next(r['exit']==0 for r in records
                                              if r.get('stage')=='tele_source_clone'))
    save(result)
    if resume:
        resume.cost(records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.ki-ocr'))
    parser.add_argument('--components', choices=['all', 'cpu', 'tele', 'paddle'], default='all')
    provisioning = parser.add_mutually_exclusive_group()
    provisioning.add_argument('--download-models', action='store_true')
    provisioning.add_argument('--preloaded-models', type=Path, help='Fixed public models: fresh empty-root install with online dependencies, or explicit offline environment reuse')
    parser.add_argument('--model-manifest', type=Path, help='Optional preload manifest; must exactly match frozen repositories/revisions/size/SHA')
    parser.add_argument('--reuse-environments', action='store_true', help='Verify all existing isolated envs and CPU imports; do not reinstall or use network')
    parser.add_argument('--native-prefix', type=Path, help='Explicit existing public native runtime for offline continuation')
    parser.add_argument('--public-pip-cache', type=Path, help='Optional cache of publicly fetched pip wheels for retry; models never use it.')
    parser.add_argument('--resume-install',action='store_true',help='One bounded continuation of a receipt-verified R5 Tele clone failure; preserve original logs')
    parser.add_argument('--resume-receipt-sha256',help='Expected SHA256 of the original root/INSTALL_RESULT.json')
    parser.add_argument('--resume-timeout-seconds',type=int,default=10800)
    args = parser.parse_args()
    kinds = ['cpu', 'tele', 'paddle'] if args.components == 'all' else [args.components]
    try:
        print(json.dumps(install(args.root, kinds, args.download_models, args.public_pip_cache,
              args.preloaded_models, args.model_manifest, args.reuse_environments, args.native_prefix,
              args.resume_install,args.resume_receipt_sha256,args.resume_timeout_seconds), indent=2))
    except Exception as error:
        if args.resume_install:
            import resume_install as continuation
            if continuation.CURRENT_RESUME is not None:
                continuation.CURRENT_RESUME.failed(error)
        failure = {'status': 'INSTALL_FAILED', 'type': type(error).__name__,
                   'error': str(error), 'GPU_started': False}
        if args.root.is_dir() and not args.preloaded_models and not args.reuse_environments and not args.resume_install:
            write(args.root / 'INSTALL_FAILURE.json', failure)
        print(json.dumps(failure))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
