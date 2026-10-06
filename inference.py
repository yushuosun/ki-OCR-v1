"""Frozen TeleOCR R3 -> current-run conditional Paddle -> official Markdown."""
import argparse
import contextlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
# -I intentionally omits script directories; add only this checked-out source.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assets import ROOT, clean_env, load, model_pins, sha, verify_pins, write

SUFFIXES = {'.png', '.jpg', '.jpeg'}


def scan_inputs(source):
    from PIL import Image
    source = Path(source).absolute()
    files = [source] if source.is_file() else sorted(source.iterdir(), key=lambda p: p.name)
    files = [p for p in files if p.suffix.lower() in SUFFIXES]
    if not files:
        raise ValueError('Input must be a PNG/JPEG page or a flat directory of PNG/JPEG pages.')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Duplicate Markdown stems; refuse overwrite or silent page removal.')
    rows = []
    for index, path in enumerate(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Input must be an ordinary image: ' + path.name)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            size = list(image.size)
        rows.append({'id': path.name, 'input_index': index, 'image_sha256': sha(path), 'size': size})
    return files[0].parent, rows


def runtime_paths(filename):
    cp = Path(filename).absolute()
    data = load(cp)
    paths = {}
    for key in ('tele_source', 'tele_model', 'paddle_root', 'tele_python', 'paddle_python', 'native_prefix'):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError('Missing runtime path: ' + key)
        # Do not resolve the executable symlink; this preserves the selected venv.
        paths[key] = Path(os.path.abspath(cp.parent / value))
    policy = data.get('execution_policy')
    if policy not in ('standalone', 'cluster'):
        raise ValueError('Explicit standalone or owner-admitted cluster execution_policy required.')
    return paths, policy


def verify_assets(paths):
    deps = load(ROOT / 'DEPENDENCIES.json')
    checks = {'tele_source': verify_pins(paths['tele_source'], deps['tele_source_pins']),
              'tele_model': verify_pins(paths['tele_model'], model_pins('tele'))}
    checks['tele_vllm_plugin'] = verify_pins(paths['tele_source'], load(ROOT / 'UPSTREAM_PLUGIN_PINS.json'))
    checks['native_library'] = verify_pins(paths['native_prefix'],load(ROOT/'NATIVE_RUNTIME_LOCK.json')['library_pins'])
    for kind in ('paddle', 'paddle_layout'):
        checks[kind] = verify_pins(paths['paddle_root'] / 'models' / kind, model_pins(kind))
    for key in ('tele_python', 'paddle_python'):
        if not paths[key].is_file():
            raise ValueError('Isolated Python missing: ' + key)
        checks[key.replace('_python','_environment')] = verify_environment(paths[key], key.removesuffix('_python'))
    verify_pins(ROOT, load(ROOT / 'CORE_SOURCE_PINS.json'))
    verify_pins(ROOT / 'archive', deps['stage_pins'])
    if sha(ROOT / 'paddle_producer.py') != deps['paddle_source_sha256']:
        raise ValueError('Frozen Paddle producer changed.')
    return checks

def check_locked_versions(actual, expected, kind):
    normalize = lambda value: value.lower().replace('_','-').replace('.','-')
    actual = {normalize(name):version for name,version in actual.items()}
    wrong = {name:{'expected':version,'actual':actual.get(normalize(name))}
             for name,version in expected.items() if actual.get(normalize(name))!=version}
    if wrong:
        priority = {'paddlepaddle-gpu':0,'paddleocr':1,'paddlex':2,'vllm':0,'torch':1,'transformers':2}
        names = sorted(wrong,key=lambda name:(priority.get(normalize(name),3),name))[:8]
        detail = {'mismatch_count':len(wrong),'examples':{name:wrong[name] for name in names}}
        raise ValueError('Frozen '+kind+' dependency missing/changed: '+json.dumps(detail,sort_keys=True))
    return len(expected)

def verify_environment(python, kind):
    config = Path(python).parent.parent / 'pyvenv.cfg'
    if not config.is_file() or 'include-system-site-packages = false' not in config.read_text(encoding='utf-8').lower():
        raise ValueError('Isolated venv required: '+kind)
    env = clean_env(config.parent);env['CUDA_VISIBLE_DEVICES']=''
    script = ("import sys,json,importlib.metadata as m; report={'prefix':sys.prefix,"
              "'base_prefix':sys.base_prefix,'python':list(sys.version_info[:3]),"
              "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}; ")
    if kind == 'paddle':
        script += "report['paddle_gpu_requires']=m.requires('paddlepaddle-gpu') if 'paddlepaddle-gpu' in {k.lower().replace('_','-') for k in report['packages']} else []; "
    script += "print(json.dumps(report))"
    metadata = json.loads(subprocess.check_output([str(python),'-I','-c',script],env=env,text=True,timeout=60))
    plugin=None
    if kind=='tele':
        script=("import sys,json; sys.path.insert(0,sys.argv[1]); from plugin_audit import inspect_installed; print(json.dumps(inspect_installed()))")
        plugin=json.loads(subprocess.check_output([str(python),'-I','-c',script,str(ROOT)],env=env,text=True,timeout=60))
    if metadata['prefix']==metadata['base_prefix'] or tuple(metadata['python'][:2])!=(3,10):
        raise ValueError('Wrong interpreter or unisolated environment: '+kind)
    expected = dict(line.strip().split('==') for line in (ROOT/('requirements-'+kind+'.lock')).read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#'))
    count = check_locked_versions(metadata['packages'],expected,kind)
    if kind == 'paddle':
        check_paddle_wheel_requirements(metadata['paddle_gpu_requires'])
    return {'exact_locked_versions_verified':count,'python':metadata['python'],'isolated_venv':True,
            'plugin_binding':plugin,'validation':'METADATA_AND_FILES_ONLY','GPU_started':False}

def check_paddle_wheel_requirements(requires):
    actual = {value.split(';')[0].strip() for value in requires if value.startswith('nvidia-')}
    expected = set(load(ROOT/'PADDLE_WHEEL_PIN.json')['CUDA_dependency_pins'])
    if actual != expected:
        raise ValueError('Paddle CUDA wheel METADATA differs from pinned cu126 build.')
    return len(expected)


def prepare(source, output, runtime, device, timeout, eager=False):
    if type(device) is not int or device < 0:
        raise ValueError('Require an explicit physical GPU index.')
    if type(timeout) is not int or timeout < 1:
        raise ValueError('Positive phase timeout required.')
    output = Path(output).absolute()
    if output.exists():
        raise ValueError('New output directory required; refusing overwrite/resume.')
    image_root, rows = scan_inputs(source)
    paths, policy = runtime_paths(runtime)
    checks = verify_assets(paths)
    deps = load(ROOT / 'DEPENDENCIES.json')
    bundle_root = paths['tele_source'] / 'TeleOCR/vlm_utils/post_process'
    verify_pins(bundle_root, deps['native_postprocessor_file_sha256'])
    bundle = {'source': 'independently downloaded pinned public Tele source',
              'files': {name: (bundle_root / name).read_text(encoding='utf-8')
                        for name in deps['native_postprocessor_file_sha256']}}
    output.mkdir(parents=True, exist_ok=False)
    harness = output / 'harness'
    shutil.copytree(ROOT / 'core', harness, ignore=shutil.ignore_patterns('__pycache__'))
    write(harness / 'NATIVE_DETERMINISTIC_CODE.json', bundle)
    write(output / 'INPUTS.json', {'reference_access': False, 'bound_rows': rows})
    contract = dict(load(ROOT / 'configs/frozen_runtime.json'),
                    candidate='R3_VLLM_COLD_SAME_BACKEND_R1',
                    selection_policy='historical_latest_abnormal', old_cache_inputs=[],
                    stage_pins=deps['stage_pins'], tele_source_pins=deps['tele_source_pins'],
                    inputs_path=str(output / 'INPUTS.json'), inputs_sha256=sha(output / 'INPUTS.json'),
                    harness=str(harness), native_bundle_sha256=sha(harness / 'NATIVE_DETERMINISTIC_CODE.json'),
                    output=str(output), execution_policy=policy,
                    phase_timeout_seconds=timeout, enforce_eager=eager,
                    runtime={'device_index': device, 'device_id': None},
                    bindings={'image_root': str(image_root), 'output_root': str(output),
                              'archive_root': str(ROOT / 'archive'),
                              'tele_source_root': str(paths['tele_source']),
                              'model_root': str(paths['tele_model']),
                              'native_prefix':str(paths['native_prefix']),
                              'tele_python': str(paths['tele_python'])},
                    paddle_binding={'root': str(paths['paddle_root']),
                                    'python': str(paths['paddle_python']),
                                    'source': str(ROOT / 'paddle_producer.py')},
                    paddle_source_sha256=deps['paddle_source_sha256'],
                    source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*')
                                   if p.is_file() and '.git' not in p.parts
                                   and not {'dependencies_local', '.ki-ocr', '__pycache__'}.intersection(p.parts)
                                   and p.suffix in {'.py', '.json', '.lock'}})
    write(output / 'CONTRACT.json', contract)
    return contract, checks


def checked_rows(contract):
    if sha(contract['inputs_path']) != contract['inputs_sha256']:
        raise ValueError('Frozen input manifest changed.')
    data = load(contract['inputs_path'])
    if set(data) != {'reference_access', 'bound_rows'} or data['reference_access'] is not False:
        raise ValueError('Unexpected or answer-bearing input manifest.')
    rows = data['bound_rows']
    if [r['input_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Frozen page order changed.')
    for row in rows:
        if (set(row) != {'id', 'input_index', 'image_sha256', 'size'}
                or Path(row['id']).name != row['id'] or '/' in row['id'] or '\\' in row['id']):
            raise ValueError('Unexpected input fields/page ID.')
        path = Path(contract['bindings']['image_root']) / row['id']
        if path.is_symlink() or sha(path) != row['image_sha256']:
            raise ValueError('Frozen input image changed: ' + row['id'])
    return rows


def load_core(contract):
    existing = sys.modules.get('entry')
    if existing is not None and Path(existing.__file__).parent != Path(contract['harness']):
        for name in ('entry', 'current_r3', 'native', 'cascade', 'paddle_contract', 'tele_support', 'pure_worker'):
            sys.modules.pop(name, None)
    sys.path[:0] = [contract['harness']]
    entry = importlib.import_module('entry')
    entry.contract_rows = lambda c, cp: checked_rows(c)
    return entry


def verify_code(contract):
    if sha(Path(contract['harness']) / 'NATIVE_DETERMINISTIC_CODE.json') != contract['native_bundle_sha256']:
        raise ValueError('Frozen native postprocessor bundle changed.')
    verify_pins(ROOT, contract['source_sha256'])
    verify_pins(contract['harness'], {Path(k).name: v for k, v in load(ROOT / 'CORE_SOURCE_PINS.json').items()})


def child_main(cp, phase):
    contract = load(cp)
    if os.environ.get('KI_OCR_CHILD_SHA') != sha(cp):
        raise ValueError('Child contract identity mismatch.')
    verify_code(contract)
    checked_rows(contract)
    verify_pins(Path(contract['bindings']['native_prefix']),load(ROOT/'NATIVE_RUNTIME_LOCK.json')['library_pins'])
    python = contract['bindings']['tele_python'] if phase == 'tele' else contract['paddle_binding']['python']
    if Path(sys.prefix).absolute() != Path(python).parent.parent:
        raise ValueError('Child did not select configured isolated venv.')
    load_core(contract)
    if contract['execution_policy'] == 'cluster':
        # Preserve the original cluster live UID/boot/lease/physical/CUDA mapping
        # checks in each isolated child, in addition to the held owner context.
        from device_compat import verify_live
        verify_live(contract['cluster_resource'])
    import pure_worker
    deadline = time.time() + contract['phase_timeout_seconds']
    if contract['execution_policy'] == 'cluster':
        deadline = min(deadline, contract['runtime'].get('owner_admission_expires_epoch', 0))
    if deadline <= time.time():
        raise ValueError('Phase deadline or actual owner lease expired.')
    contract['runtime']['phase_deadline_epoch'] = deadline
    run = Path(contract['output'])
    if phase == 'tele':
        import vllm
        from plugin_audit import import_and_audit,audit_loaded
        write(run/'PLUGIN_SOURCE_BEFORE_MODEL.json',import_and_audit())
        if contract.get('max_generation_items') is not None:
            from generation_budget import GenerationBudget, hook_tele
            hook_tele(vllm.LLM, GenerationBudget(run, contract['max_generation_items']),
                      contract['enforce_eager'], lambda value: write(run / 'GENERATION_RUNTIME.json', value))
        original = vllm.LLM.__init__

        def init(llm, *args, **kwargs):
            kwargs['enforce_eager'] = contract['enforce_eager']
            original(llm, *args, **kwargs)
            write(run/'PLUGIN_SOURCE_AFTER_MODEL.json',audit_loaded())
            actual = llm.llm_engine.vllm_config.model_config
            if actual.seed != 0 or actual.max_model_len != 16384:
                raise ValueError('Actual seed/context differs from frozen method.')
            write(run / 'ACTUAL_RUNTIME.json', {'seed': actual.seed, 'context': actual.max_model_len,
                                               'enforce_eager': actual.enforce_eager})
        vllm.LLM.__init__ = init
        pure_worker.tele_phase(contract, Path(cp), run)
    else:
        if contract.get('max_generation_items') is not None:
            from generation_budget import GenerationBudget, hook_paddle
            from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
            hook_paddle(DocVLMLocalPredictor, GenerationBudget(run, contract['max_generation_items']))
        pure_worker.paddle_phase(contract, Path(cp), run if phase == 'paddle' else run / 'PADDLE_PROBE')


def child(contract, cp, phase):
    python = contract['bindings']['tele_python'] if phase == 'tele' else contract['paddle_binding']['python']
    env = clean_env(Path(contract['output']) / 'runtime-cache',Path(contract['bindings']['native_prefix'])/'lib')
    env.update(CUDA_VISIBLE_DEVICES=str(contract['runtime']['device_index']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', KI_OCR_CHILD_SHA=sha(cp))
    if contract['execution_policy'] == 'cluster':
        entry = load_core(contract)
        from device_compat import numeric_child_env
        resource = contract['cluster_resource']
        env['CUDA_VISIBLE_DEVICES'] = resource['gpu_uuid']
        env = numeric_child_env(resource, env)
    started = time.monotonic()
    remaining = contract['phase_timeout_seconds']
    if contract['execution_policy'] == 'cluster':
        remaining = min(remaining, contract['runtime'].get('owner_admission_expires_epoch', 0) - time.time())
    if remaining <= 0:
        raise ValueError('Actual owner lease expired before child launch.')
    logfile = Path(contract['output']) / (phase + '.log')
    memory_samples, sample_errors, failure_reason = [], 0, None
    with logfile.open('xb') as log:
        process = subprocess.Popen([python, '-I', '-B', str(ROOT / 'inference.py'), '_child',
                                    '--contract', str(cp), '--phase', phase], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + remaining
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, remaining)
                try:
                    sample = subprocess.check_output(['nvidia-smi', '-i', str(contract['runtime']['device_index']),
                             '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
                    identity, used = sample.strip().split(',')
                    if identity.strip() != contract['runtime']['device_id']:
                        raise RuntimeError('Selected GPU identity changed during owned phase.')
                    memory_samples.append(int(used))
                except (subprocess.SubprocessError, ValueError, OSError):
                    sample_errors += 1
                try:
                    process.wait(timeout=min(2, max(0.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            code = process.returncode
            timed_out = False
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            # Stop only this newly created child process group.
            failure_reason = str(error)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code, timed_out = process.returncode or 2, isinstance(error, subprocess.TimeoutExpired)
    return {'exit': code, 'timed_out': timed_out, 'failure_reason': failure_reason, 'wall_seconds': time.monotonic() - started,
            'log': logfile.name, 'log_sha256': sha(logfile),
            'selected_gpu_peak_sampled_mib': max(memory_samples) if memory_samples else None,
            'memory_samples': len(memory_samples), 'memory_sample_errors': sample_errors,
            'memory_note': 'Device sampling every ~2s; may miss transient peaks and is not tensor allocated memory.'}


def pipeline(contract, cp, driver=None, probe=False):
    """A driver is allowed only for explicit synthetic CPU tests, never CLI inference."""
    entry = load_core(contract)
    from current_r3 import compose
    from paddle_contract import validate_donor
    run, rows = Path(contract['output']), checked_rows(contract)
    synthetic = driver is not None
    invoke = driver or (lambda phase: child(contract, cp, phase))
    phases = {}
    started = time.monotonic()
    try:
        phases['tele'] = invoke('tele')
    except Exception as error:
        phases['tele'] = {'exit': 2, 'error': str(error)}
    records = entry.jsonl(run / 'PRIVATE_RAW_REQUESTS.jsonl')
    adoptions = entry.jsonl(run / 'ADOPTIONS.jsonl')
    pins = {stage: contract['stage_pins'][name] for stage, name in entry.STAGES.items()}
    pages, eligible, page_errors = {}, [], {}
    for index, item in enumerate(rows):
        row = dict(item, page_id=Path(item['id']).stem, local_index=index)
        try:
            md, middle = entry.fresh_page(run, row)
            pages[row['page_id']] = (row, md, middle)
            if md is not None and compose(row, md, middle, records, adoptions, sha(cp), pins)[2]:
                eligible.append({'page_id': row['page_id'],
                                 'image_path': str(Path(contract['bindings']['image_root']) / row['id']),
                                 'input_sha256': row['image_sha256']})
        except Exception as error:
            page_errors[row['page_id']] = str(error)
            pages[row['page_id']] = (row, None, None)
    write(run / 'PADDLE_INPUTS.json', {'pages': eligible, 'reference_access': False})
    if eligible:
        try:
            phases['paddle'] = invoke('paddle')
        except Exception as error:
            phases['paddle'] = {'exit': 2, 'error': str(error)}
    else:
        phases['paddle'] = {'status': 'NOT_TRIGGERED', 'exit': 0}
    terminals = []
    for page_id, (row, md, middle) in pages.items():
        final, decisions = md or '', []
        state = 'COMPLETE' if md is not None else 'MISSING_RETAINED'
        if md is not None:
            try:
                folder = run / 'PADDLE' / page_id
                donor, donor_md = None, None
                if (folder / 'native.json').is_file():
                    donor = load(folder / 'native.json')
                    donor = donor.get('res', donor)
                    validate_donor(donor, middle['pdf_info'][0]['page_size'])
                    donor_md = (folder / 'prediction.md').read_text(encoding='utf-8')
                final, decisions, trigger = compose(row, md, middle, records, adoptions, sha(cp), pins, donor, donor_md)
                if trigger and donor is None:
                    state = 'FALLBACK_FAILED_RETAINED'
            except Exception as error:
                final, state = md, 'FALLBACK_FAILED_RETAINED'
                page_errors[page_id] = str(error)
                decisions = [{'action': 'retain', 'error': str(error)}]
        if not final.strip():
            state = 'EMPTY_RETAINED'
        target = run / 'markdown' / (page_id + '.md')
        target.parent.mkdir(exist_ok=True)
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(final)
        terminals.append({'id': row['id'], 'state': state, 'in_denominator': True,
                          'prediction_sha256': sha(target), 'decisions': decisions,
                          'error': page_errors.get(page_id)})
    probe_result = {'status': 'NOT_RUN', 'natural_fallback': False}
    if probe and phases['tele']['exit'] == 0 and not eligible and rows:
        directory = run / 'PADDLE_PROBE'
        directory.mkdir()
        row = rows[0]
        page_id = Path(row['id']).stem
        write(directory / 'PADDLE_INPUTS.json', {'pages': [{'page_id': page_id,
              'image_path': str(Path(contract['bindings']['image_root']) / row['id']),
              'input_sha256': row['image_sha256']}], 'reference_access': False})
        try:
            probe_result.update(invoke('paddle_probe'))
            if probe_result.get('exit', 0) != 0:
                raise RuntimeError('Paddle probe child failed; partial donor artifacts do not pass.')
            donor = load(directory / 'PADDLE' / page_id / 'native.json')
            donor = donor.get('res', donor)
            validate_donor(donor, row['size'])
            donor_md = (directory / 'PADDLE' / page_id / 'prediction.md').read_text(encoding='utf-8')
            final, _, trigger = compose(dict(row, page_id=page_id), '',
                      {'pdf_info': [{'page_size': row['size'], 'para_blocks': []}]},
                      [], [], sha(cp), pins, donor, donor_md)
            if not final.strip() or not trigger:
                raise ValueError('Controlled empty-primary Paddle merge did not produce output.')
            probe_result.update(status='CONTROLLED_DONOR_PROBE', serialization_pass=True,
                                synthetic_empty_primary_merge_pass=bool(final.strip()) and trigger)
        except Exception as error:
            probe_result.update(status='PROBE_FAILED', error=str(error))
    failed = sum(t['state'] != 'COMPLETE' for t in terminals)
    phase_failed = any(p.get('exit', 0) != 0 for p in phases.values())
    overall_success = not failed and not phase_failed and probe_result.get('status') != 'PROBE_FAILED'
    result = {'status': 'CPU_SYNTHETIC_ONLY' if synthetic else
              'INFERENCE_COMPLETE' if overall_success else 'INFERENCE_WITH_ERRORS',
              'overall_success':overall_success,
              'synthetic': synthetic, 'pages': len(rows), 'returned_pages': len(rows) - failed,
              'failed_retained': failed, 'natural_fallback_pages': len(eligible),
              'phases': phases, 'terminals': terminals, 'paddle_probe': probe_result,
              'tele_length_stops': sum(q.get('finish_reason') == 'length' for q in records),
              'GT_read': False, 'official_score': None, 'input_manifest_sha256': contract['inputs_sha256'],
              'contract_sha256': sha(cp), 'method': 'NATIVE->FORMULA125->GUARD->TABLE1X',
              'historical_98_38731_reproduced': False}
    wall = time.monotonic() - started
    cost = {'wall_seconds': wall, 'wall_seconds_per_input_page': wall / len(rows),
            'GPU_card_seconds': 0 if synthetic else None,
            'GPU_cost_note': 'NOT_RUN' if synthetic else 'Requires allocation telemetry; phase wall includes model load and CPU work.',
            'auto_retry': False, 'phases': phases}
    if (run / 'TELE_RESULT.json').is_file():
        cost['tele_runtime'] = load(run / 'TELE_RESULT.json')
    write(run / 'RESULT.json', result)
    write(run / 'COST.json', cost)
    return result


@contextlib.contextmanager
def standalone_device(contract):
    import fcntl
    gpu = str(contract['runtime']['device_index'])
    observed = subprocess.run(['nvidia-smi', '-i', gpu,
               '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
               capture_output=True, text=True, timeout=15, check=True).stdout.strip().split(',')
    if len(observed) != 3 or int(observed[1]) != 0 or int(observed[2]) != 0:
        raise ValueError('Selected GPU is busy or telemetry unknown; no process was stopped.')
    contract['runtime']['device_id'] = observed[0].strip()
    lock_root = Path(contract['bindings']['tele_python']).parent.parent.parent / 'device-locks'
    lock_root.mkdir(exist_ok=True)
    with (lock_root / (observed[0].strip() + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_child':
        parser = argparse.ArgumentParser()
        parser.add_argument('--contract', type=Path, required=True)
        parser.add_argument('--phase', choices=['tele', 'paddle', 'paddle_probe'], required=True)
        args = parser.parse_args(sys.argv[2:])
        child_main(args.contract, args.phase)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=Path('.ki-ocr/runtime.json'))
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--phase-timeout', type=int, default=86400)
    parser.add_argument('--eager', action='store_true', help='Explicit runtime workaround; recorded in contract.')
    parser.add_argument('--paddle-probe', action='store_true', help='Controlled real donor probe if natural fallback absent; does not alter output.')
    args = parser.parse_args()
    if args.action == 'run' and sys.platform != 'linux':
        raise ValueError('GPU inference requires Linux.')
    contract, checks = prepare(args.input, args.output, args.runtime, args.device, args.phase_timeout, args.eager)
    cp = args.output.absolute() / 'CONTRACT.json'
    if args.action == 'check':
        print(json.dumps({'status': 'CPU_INPUT_ASSET_CHECK_PASS', 'pages': len(checked_rows(contract)),
                          'environment_validation':'METADATA_AND_FILES_ONLY_NOT_INFERENCE_READY',
                          'checks': checks, 'GPU_started': False, 'contract': str(cp)}))
        return
    if contract['execution_policy'] == 'cluster':
        raise ValueError('Cluster execution requires the original owner admission adapter; this public CLI cannot admit or register jobs.')
    with standalone_device(contract):
        write(cp, contract)
        result = pipeline(contract, cp, probe=args.paddle_probe)
    # Metadata only. Prediction/raw answer bodies remain in the user's local output.
    print(json.dumps({k: v for k, v in result.items() if k not in {'terminals'}}, indent=2))
    if result['status'] != 'INFERENCE_COMPLETE' or result['paddle_probe'].get('status') == 'PROBE_FAILED':
        raise SystemExit(2)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'INFERENCE_BLOCKED_OR_FAILED', 'type': type(error).__name__, 'error': str(error)}))
        raise SystemExit(2)
