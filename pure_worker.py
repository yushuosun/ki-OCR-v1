"""Pure inference functions derived from the frozen project entry.

Only host plumbing differs: explicit device ID and phase deadline replace the
cluster owner lease fields. Pinned assets are checked by inference.prepare.
The original core/entry.py and all cluster admission functions remain unchanged.
No registration, lease acquisition, cluster path or cluster identity is used here.
"""
import hashlib, json, os, signal, sys, time
from pathlib import Path
from entry import HERE, STAGES, AcceptedWrites, append, jsonl, load, materialize, owned, put, require, sha, stage_code, stage_input_rows, stage_plan, utc
from inference import checked_rows

def contract_rows(c, cp):
    return checked_rows(c)

def tele_phase(c, cp, run):
    import threading
    import tele_support as support
    from PIL import Image
    bindings = c['bindings']
    raw_rows = contract_rows(c, cp)
    rows = []
    for r in raw_rows:
        p = Path(bindings['image_root']) / r['id']
        require(sha(p) == r['image_sha256'], 'CURRENT_INPUT_CHANGED')
        with Image.open(p) as im:
            size = list(im.size)
        rows.append(dict(r, page_id=Path(r['id']).stem, image=str(p), size=size))
    pins = {str((Path(bindings['archive_root']) / n).resolve()): h for n, h in c['stage_pins'].items()}
    for p, h in pins.items():
        require(sha(p) == h, 'ORIGINAL_STAGE_PIN_CHANGED')
    source_root = Path(bindings['tele_source_root']).resolve()
    for name, h in c['tele_source_pins'].items():
        p = (source_root / name).resolve()
        require(p.is_relative_to(source_root) and sha(p) == h, 'TELE_SOURCE_PIN_CHANGED:' + name)
    sys.path[:0] = [bindings['archive_root'], bindings['tele_source_root'], *bindings.get('extra_python_paths', [])]
    import TeleOCR.config as config
    config.MAX_MODEL_LEN = 16384
    config.GPU_MEMORY_UTILIZATION = 0.5
    os.environ['TELE_GPU_UTIL'] = '0.5'
    import vllm
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    client = TeleOCRMODEL_SERVICE.get_model('vllm-engine', bindings['model_root'], None)
    require(client.client.model_max_length == 16384 and client.client.batch_size == 0, 'ACTUAL_CONTEXT_OR_BATCHING_CHANGED')
    require(client.batching_mode == 'stepping', 'ORIGINAL_STEPPING_REQUIRED')
    require(support.plain(client.helper.sampling_params) == c['expected_sampling'], 'FROZEN_PRESENCE_FREQUENCY_AND_SAMPLING_CHANGED')
    require(support.plain(client.helper.prompts) == c['expected_prompts'], 'FROZEN_PROMPTS_CHANGED')

    def verify_loaded_sources():
        from source_audit import audit
        result = audit(source_root, c['tele_source_pins'], sys.modules)
        append(run / 'SOURCE_AUDITS.jsonl', dict(result, utc=utc()))
        require(result['status'] == 'PASS', 'LOADED_SOURCE_AUDIT_REJECTED:' + json.dumps(result['errors']))
    verify_loaded_sources()
    put(run / 'LOADED.json', {'backend': 'vllm-engine', 'max_model_len': client.client.model_max_length, 'model': bindings['model_root'], 'runtime_versions': {'vllm': vllm.__version__, 'torch': __import__('torch').__version__, 'transformers': __import__('transformers').__version__}, 'sampling': support.plain(client.helper.sampling_params), 'historical_exact_reproduction': 'NOT_PROVED_BY_SAME_BACKEND', 'device_id': c['runtime']['device_id'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'loaded_utc': utc()})

    class Budget:

        def __init__(self):
            self.submitted = 0
            self.deadline = c['runtime']['phase_deadline_epoch']

        def remaining(self):
            remaining = self.deadline - time.time()
            require(remaining > 0, 'REQUEST_DEADLINE_EXPIRED')
            return remaining

        def reserve(self, n):
            self.submitted += n
            return min(600, self.remaining())
    w = support.Worker.__new__(support.Worker)
    w.contract = c
    w.contract_path = Path(cp)
    w.run_contract_sha256 = sha(cp)
    w.worker_sha256 = sha(HERE / 'entry.py')
    w.job = run
    w.tele = {'archive_root': bindings['archive_root']}
    w.rows = rows
    w.by_id = {r['page_id']: r for r in rows}
    w.pins = pins
    w.config = config
    w.budget = Budget()
    w.started = time.monotonic()
    w.local = threading.local()
    w.preparation_events = []
    w.preparation_lock = threading.Lock()
    w.state = {'stage': 'LOADING', 'batch': None, 'calls': 0, 'completed_requests': 0, 'completed_pages': []}
    w.bindings = []
    w.page_ids = []
    w.crop_candidates = {}

    def timeout(sig, frame):
        raise TimeoutError('REQUEST_600S_OR_PHASE_DEADLINE_EXPIRED')
    signal.signal(signal.SIGALRM, timeout)
    w.install_observers(client, vllm.LLM)
    stage_raw = []
    original_append = support.append

    def observed_append(path, value):
        original_append(path, value)
        if Path(path).name == 'PRIVATE_RAW_REQUESTS.jsonl':
            stage_raw.append(value)
    support.append = observed_append
    adoption_ordinal = 0
    for batch, pos in enumerate(range(0, len(rows), 16)):
        batch_rows = rows[pos:pos + 16]
        d = run / 'BATCHES' / f'{batch:04d}'
        d.mkdir(parents=True, exist_ok=False)
        put(d / 'INPUTS.json', {'bound_rows': stage_input_rows(batch_rows)})
        w.state['batch'] = batch
        w.page_ids = [r['page_id'] for r in batch_rows]
        object_rows = {}

        def bind_object(obj, kind, row):
            object_rows[id(obj)] = Path(row['id']).stem

        def accept(obj, kind, txt):
            nonlocal adoption_ordinal
            bbox = list(obj['bbox'])
            qlist = [q for q in stage_raw if q['binding'].get('bbox') == bbox and q['binding'].get('kind') == kind]
            page_id = object_rows.get(id(obj))
            qlist = [q for q in qlist if q['binding'].get('page_id') == page_id]
            q = max(qlist, key=lambda q: (q['call'], q['batch_index'])) if qlist else None
            adoption_ordinal += 1
            row_identity = w.by_id.get(page_id, {})
            append(run / 'ADOPTIONS.jsonl', {'ordinal': adoption_ordinal, 'stage': w.state['stage'], 'page_id': page_id, 'input_image_sha256': row_identity.get('image_sha256'), 'kind': kind, 'bbox': bbox, 'accepted_payload_sha256': hashlib.sha256(txt.encode()).hexdigest(), 'call': q['call'] if q else None, 'batch_index': q['batch_index'] if q else None, 'run_contract_sha256': w.run_contract_sha256, 'accepted_write_observed': True})
        try:
            for stage, args in stage_plan(d):
                if stage == 'TABLE1X':
                    materialize(d, batch_rows, 'GUARD_VIEW', ['GUARD', 'FORMULA125', 'NATIVE'])
                w.state['stage'] = stage
                w.bindings = []
                w.active_script = str(Path(bindings['archive_root']) / STAGES[stage])
                require(sha(w.active_script) == c['stage_pins'][STAGES[stage]], 'STAGE_CODE_CHANGED_DURING_RUN')
                require(sha(cp) == w.run_contract_sha256, 'CONTRACT_CHANGED_DURING_RUN')
                stage_raw.clear()
                if stage != 'NATIVE':
                    w.capture_crop_candidates(d, batch_rows, stage)
                append(run / 'STAGES.jsonl', {'stage': stage, 'batch': batch, 'status': 'START', 'utc': utc()})
                previous = sys.argv
                try:
                    sys.argv = [w.active_script, '--contract', str(d / 'INPUTS.json'), '--model', bindings['model_root'], *args]
                    object_rows.clear()
                    exec(stage_code(w.active_script), {'__name__': '__main__', '__file__': w.active_script, '__r3_accept__': accept, '__r3_bind__': bind_object})
                finally:
                    sys.argv = previous
                append(run / 'STAGES.jsonl', {'stage': stage, 'batch': batch, 'status': 'COMPLETE', 'utc': utc()})
            materialize(d, batch_rows, 'B0', ['TABLE1X', 'GUARD_VIEW'])
            for row in batch_rows:
                append(run / 'PAGES.jsonl', {'page_id': row['page_id'], 'image_sha256': row['image_sha256'], 'size': row['size'], 'batch': batch, 'status': 'COMPLETE'})
        except Exception as exc:
            append(run / 'ERRORS.jsonl', {'batch': batch, 'stage': w.state['stage'], 'error': type(exc).__name__, 'message': str(exc), 'pages': [r['page_id'] for r in batch_rows]})
            if isinstance(exc, (MemoryError, TimeoutError)) or 'out of memory' in str(exc).lower():
                raise
    signal.setitimer(signal.ITIMER_REAL, 0)
    verify_loaded_sources()
    import torch
    put(run / 'TELE_RESULT.json', {'pages_completed': len(jsonl(run / 'PAGES.jsonl')), 'submitted_requests': w.budget.submitted, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'actual_gpu_phase': True, 'wall_seconds': time.monotonic() - w.started})

def paddle_phase(c, cp, run):
    """Invoke exact pinned producer worker with fresh current-only manifest."""
    import runpy
    from types import SimpleNamespace
    p = c['paddle_binding']
    require(sha(p['source']) == c['paddle_source_sha256'], 'PADDLE_PRODUCER_PIN_CHANGED')
    output = run / 'PADDLE'
    output.mkdir(exist_ok=False)
    import importlib.metadata
    from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
    original_build = DocVLMLocalPredictor._build
    requests = 0
    phase_start = time.monotonic()

    def timeout(sig, frame):
        raise TimeoutError('PADDLE_REQUEST_600S_OR_PHASE_DEADLINE_EXPIRED')
    signal.signal(signal.SIGALRM, timeout)

    def observed_build(self, **kwargs):
        append(run / 'PADDLE_MODEL_LOADS.jsonl', {'status': 'START', 'utc': utc()})
        try:
            model, processor = original_build(self, **kwargs)
        except Exception as exc:
            append(run / 'PADDLE_MODEL_LOADS.jsonl', {'status': 'ERROR', 'error': type(exc).__name__, 'message': str(exc), 'utc': utc()})
            raise
        append(run / 'PADDLE_MODEL_LOADS.jsonl', {'status': 'COMPLETE', 'model_type': type(model).__name__, 'processor_type': type(processor).__name__, 'utc': utc()})
        generate = model.generate

        def observed_generate(*args, **options):
            nonlocal requests
            ids = options.get('input_ids')
            count = int(ids.shape[0]) if ids is not None else 1
            require(options.get('max_new_tokens') == 4096, 'PINNED_PADDLE_OUTPUT_CAP_CHANGED')
            remaining = c['runtime']['phase_deadline_epoch'] - time.time()
            require(remaining > 0, 'REQUEST_DEADLINE_EXPIRED')
            requests += count
            append(run / 'PADDLE_REQUESTS.jsonl', {'requests_so_far': requests, 'input_shape': list(ids.shape) if ids is not None else None, 'actual_attention_mask_tokens': int(options['attention_mask'].sum().item()) if options.get('attention_mask') is not None else None, 'max_new_tokens': options.get('max_new_tokens'), 'processor': type(processor).__name__, 'utc': utc()})
            signal.setitimer(signal.ITIMER_REAL, min(600, remaining))
            try:
                return generate(*args, **options)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        model.generate = observed_generate
        return (model, processor)
    DocVLMLocalPredictor._build = observed_build
    module = runpy.run_path(p['source'], run_name='r3_pinned_paddle_producer')
    require(callable(module.get('worker')), 'PINNED_PRODUCER_WORKER_MISSING')
    module['worker'](SimpleNamespace(root=p['root'], output=str(output), manifest=str(run / 'PADDLE_INPUTS.json')))
    put(run / 'PADDLE_RESULT.json', {'actual_gpu_phase': True, 'requests': requests, 'wall_seconds': time.monotonic() - phase_start, 'runtime_versions': {n: importlib.metadata.version(n) for n in ['paddleocr', 'paddlex', 'paddlepaddle-gpu']}, 'source_sha256': sha(p['source'])})
