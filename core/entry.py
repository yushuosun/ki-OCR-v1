"""Executable cache-free R3 supervisor. Parent registers/submits; never self-registers.

run: Tele subprocess -> same-run trigger -> pinned Paddle subprocess -> R3.
check: CPU contract inspection only. No model import occurs outside tele/paddle.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STAGES = {'NATIVE':'run_tele_img.py','FORMULA125':'stack_reread_v1.py','GUARD':'stack_guard_v1.py','TABLE1X':'stack_reread_v1.py'}
INPUT_SHA = '363cdb89f7ba01e742474454011e36e57d9601566baf4660ce94ef52d7b711c8'
FORMAL_FILES={'entry.py','current_r3.py','paddle_contract.py','tele_support.py','cascade.py','native.py',
    'NATIVE_DETERMINISTIC_CODE.json','device_compat.py','source_audit.py','formal_gates.py','fresh_cost.py','formal_terminal.py','merge_formal.py'}


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(ok, reason):
    if not ok: raise ValueError(reason)
def put(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def append(p,value):
    with Path(p).open('a',encoding='utf-8') as f: f.write(json.dumps(value,ensure_ascii=False,default=str)+'\n')
def jsonl(p):
    p=Path(p)
    if not p.exists(): return []
    result=[]
    for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
        if not line.strip(): continue
        try:
            value=json.loads(line)
            if isinstance(value,dict): result.append(value)
        except ValueError:
            # A killed writer may leave one partial record. It cannot bind raw.
            continue
    return result
def owned(root,p):
    root,p=Path(root).resolve(),Path(p).resolve()
    require(p.is_relative_to(root),'PATH_ESCAPES_NEW_RUN')
    return p


def contract_rows(c, cp):
    from current_r3 import POLICY_CANDIDATES,HISTORICAL
    require(c['candidate']==POLICY_CANDIDATES.get(c.get('selection_policy',HISTORICAL)),'CANDIDATE_SELECTION_POLICY_MISMATCH')
    require(c['backend']=='vllm-engine' and c['max_model_len']==16384,'FROZEN_BACKEND_CONTEXT')
    require(c['old_cache_inputs']==[],'OLD_CACHE_INPUTS_FORBIDDEN')
    require(c['source_manifest_sha256']==INPUT_SHA,'ORIGINAL_MANIFEST_IDENTITY')
    require(c['scope'] in ('smoke','gpu2','gpu4'),'FROZEN_SCOPE')
    if c.get('experiment_kind')=='FRESH_FORMAL_RUN':
        from formal_gates import fresh_contract
        fresh_contract(c)
    p=Path(cp).parent/c['inputs']
    require(p.parent.resolve()==Path(cp).parent.resolve(),'INPUT_MANIFEST_MUST_BE_SIBLING')
    require(sha(p)==c['inputs_sha256'],'INPUT_MANIFEST_CHANGED')
    rows=load(p)['bound_rows']
    count={'smoke':1,'gpu2':826,'gpu4':825}[c['scope']]
    start=826 if c['scope']=='gpu4' else 0
    require(len(rows)==len({r['id'] for r in rows})==count,'FIXED_ROSTER_CARDINALITY')
    require([r['original_roster_index'] for r in rows]==list(range(start,start+count)),'FIXED_ROSTER_ORDER')
    require(len({Path(r['id']).stem for r in rows})==count,'STEM_COLLISION')
    require(all(Path(r['id']).name==r['id'] and '/' not in r['id'] and '\\' not in r['id'] for r in rows),'UNSAFE_PAGE_ID')
    return rows


def validate_device_binding(c):
    fixed=c.get('fixed_gpu_index')
    actual=c['resource'].get('gpu_index')
    require(type(fixed) is int and 1<=fixed<=6,'FIXED_GPU_MUST_BE_1_TO_6')
    require(type(actual) is int and 1<=actual<=6,'RESOURCE_GPU_MUST_BE_1_TO_6')
    require(actual==fixed,'FROZEN_GPU_CONTRACT_MISMATCH')


def validate(c,cp,live=False):
    rows=contract_rows(c,cp)
    formal=c.get('experiment_kind')=='FRESH_FORMAL_RUN'
    unassigned=formal and c.get('fixed_gpu_index') is None and c['resource'].get('gpu_index') is None
    if unassigned:require(not live,'ACTUAL_SECOND_GPU_ASSIGNMENT_REQUIRED')
    else:validate_device_binding(c)
    if formal:require(set(c['entry_pins'])==FORMAL_FILES,'ALL_FINAL_CODE_DEPENDENCIES_REQUIRED')
    for name in STAGES.values(): require(len(c['stage_pins'].get(name,''))==64,'MISSING_STAGE_PIN')
    for name,h in c['entry_pins'].items():
        p=HERE/name
        require(p.resolve().parent==HERE and sha(p)==h,'ENTRY_CODE_CHANGED:'+name)
    if live:
        require(sys.platform=='linux','LIVE_REQUIRES_LINUX')
        b=c['bindings']; resource=c['resource']
        for key in ['tele_python','tele_source_root','archive_root','model_root','image_root','output_root']:
            require(isinstance(b.get(key),str) and Path(b[key]).is_absolute(),'BINDING_REQUIRED:'+key)
        require(resource.get('parent_admitted') is True and resource.get('owner_lease_verified') is True,'PARENT_RESOURCE_ADMISSION_REQUIRED')
        require(isinstance(resource.get('gpu_uuid'),str) and resource['gpu_uuid'].startswith('GPU-') and ',' not in resource['gpu_uuid'],'SINGLE_GPU_UUID_REQUIRED')
        require(time.time()<resource.get('lease_expires_epoch',0),'RESOURCE_LEASE_EXPIRED')
        require(resource.get('assets_verified_manifest_sha256')==c['asset_pin_manifest_sha256'],'CURRENT_ASSET_IDENTITY_REQUIRED')
        if formal:
            from formal_gates import acceptance
            acceptance(c)
        elif c['scope']!='smoke':
            require(c.get('smoke_pass_contract_sha256') and c.get('independent_review_pass') is True,'FORMAL1651_SMOKE_AND_INDEPENDENT_REVIEW_REQUIRED')
        require(not Path(b['output_root']).exists(),'NEW_EMPTY_RUN_ROOT_REQUIRED_NO_RESUME')
    return rows


class AcceptedWrites(ast.NodeTransformer):
    """Observe original accepting assignments; do not alter their conditions/RHS."""
    def __init__(self): self.hooks=0
    def visit_For(self,node):
        self.generic_visit(node)
        if isinstance(node.target,ast.Tuple):
            names=[x.id if isinstance(x,ast.Name) else '' for x in node.target.elts]
            if names in (['sp','k','key'],['obj','kind','key']):
                var,kind=names[:2]
                hook=ast.parse('__r3_bind__('+var+', '+kind+', r)').body[0]
                node.body.insert(0,ast.copy_location(hook,node))
        return node
    def visit_Assign(self,node):
        if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Subscript): return node
        t=node.targets[0]
        if not isinstance(t.value,ast.Name): return node
        var=t.value.id
        match=(var=='sp' and isinstance(t.slice,ast.Name) and t.slice.id=='key' and isinstance(node.value,ast.Name) and node.value.id=='txt')
        match=match or (var=='obj' and ((isinstance(t.slice,ast.Name) and t.slice.id=='key' and isinstance(node.value,ast.Name) and node.value.id=='txt') or (isinstance(t.slice,ast.Constant) and t.slice.value=='lines')))
        if not match: return node
        self.hooks+=1
        hook=ast.parse('__r3_accept__('+var+', '+('k' if var=='sp' else 'kind')+', txt)').body[0]
        return [node,ast.copy_location(hook,node)]


def stage_code(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8')); observer=AcceptedWrites()
    tree=observer.visit(tree); ast.fix_missing_locations(tree)
    expected={'run_tele_img.py':0,'stack_reread_v1.py':1,'stack_guard_v1.py':2}[Path(path).name]
    require(observer.hooks==expected,'ORIGINAL_ACCEPTANCE_AST_CHANGED')
    return compile(tree,str(path),'exec')


def stage_plan(d):
    d=Path(d)
    return [
        ('NATIVE',['--out',str(d/'NATIVE'),'--chunk','16']),
        ('FORMULA125',['--run',str(d/'NATIVE'),'--out',str(d/'FORMULA125'),'--kinds','equation','--scale','1.25','--chunk','32']),
        ('GUARD',['--run',str(d/'FORMULA125'),'--fallback',str(d/'NATIVE'),'--out',str(d/'GUARD'),'--scales','1.0,0.75,1.5']),
        ('TABLE1X',['--run',str(d/'GUARD_VIEW'),'--out',str(d/'TABLE1X'),'--kinds','table','--scale','1.0','--chunk','32'])]


def stage_input_rows(rows):
    for row in rows:
        require(Path(row['id']).stem==row['page_id'],'STAGE_INPUT_ID_ALREADY_STRIPPED_OR_MISMATCH')
    return rows


def materialize(d,rows,name,priority):
    """Sparse stage fallback is restricted to this batch's freshly created dirs."""
    import shutil
    for sub,ext in [('markdown','md'),('middle','json')]:
        dest=owned(d,Path(d)/name/sub); dest.mkdir(parents=True,exist_ok=False)
        for r in rows:
            filename=r['page_id']+'.'+ext
            source=next((owned(d,Path(d)/s/sub/filename) for s in priority if owned(d,Path(d)/s/sub/filename).is_file()),None)
            require(source is not None,'MISSING_FRESH_STAGE_ARTIFACT:'+filename)
            shutil.copyfile(source,dest/filename)


def tele_phase(c,cp,run):
    import threading
    import tele_support as support
    from PIL import Image
    bindings=c['bindings']; raw_rows=contract_rows(c,cp)
    rows=[]
    for r in raw_rows:
        p=Path(bindings['image_root'])/r['id']
        require(sha(p)==r['image_sha256'],'CURRENT_INPUT_CHANGED')
        with Image.open(p) as im: size=list(im.size)
        rows.append(dict(r,page_id=Path(r['id']).stem,image=str(p),size=size))
    pins={str((Path(bindings['archive_root'])/n).resolve()):h for n,h in c['stage_pins'].items()}
    for p,h in pins.items(): require(sha(p)==h,'ORIGINAL_STAGE_PIN_CHANGED')
    source_root=Path(bindings['tele_source_root']).resolve()
    for name,h in c['tele_source_pins'].items():
        p=(source_root/name).resolve(); require(p.is_relative_to(source_root) and sha(p)==h,'TELE_SOURCE_PIN_CHANGED:'+name)
    sys.path[:0]=[bindings['archive_root'],bindings['tele_source_root'],*bindings.get('extra_python_paths',[])]
    import TeleOCR.config as config
    config.MAX_MODEL_LEN=16384; config.GPU_MEMORY_UTILIZATION=0.5
    os.environ['TELE_GPU_UTIL']='0.5'
    import vllm
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    # Preserve frozen factory/sampling; no HF route and no custom cap/penalty tuning.
    client=TeleOCRMODEL_SERVICE.get_model('vllm-engine',bindings['model_root'],None)
    require(client.client.model_max_length==16384 and client.client.batch_size==0,'ACTUAL_CONTEXT_OR_BATCHING_CHANGED')
    require(client.batching_mode=='stepping','ORIGINAL_STEPPING_REQUIRED')
    require(support.plain(client.helper.sampling_params)==c['expected_sampling'],'FROZEN_PRESENCE_FREQUENCY_AND_SAMPLING_CHANGED')
    require(support.plain(client.helper.prompts)==c['expected_prompts'],'FROZEN_PROMPTS_CHANGED')
    def verify_loaded_sources():
        from source_audit import audit
        result=audit(source_root,c['tele_source_pins'],sys.modules)
        append(run/'SOURCE_AUDITS.jsonl',dict(result,utc=utc()))
        require(result['status']=='PASS','LOADED_SOURCE_AUDIT_REJECTED:'+json.dumps(result['errors']))
    verify_loaded_sources()
    put(run/'LOADED.json',{'backend':'vllm-engine','max_model_len':client.client.model_max_length,
        'model':bindings['model_root'],'runtime_versions':{'vllm':vllm.__version__,'torch':__import__('torch').__version__,'transformers':__import__('transformers').__version__},
        'sampling':support.plain(client.helper.sampling_params),'historical_exact_reproduction':'NOT_PROVED_BY_SAME_BACKEND',
        'gpu_uuid':c['resource']['gpu_uuid'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'loaded_utc':utc()})
    class Budget:
        def __init__(self): self.submitted=0; self.deadline=c['resource']['lease_expires_epoch']
        def remaining(self):
            remaining=self.deadline-time.time(); require(remaining>0,'OWNER_LEASE_EXPIRED'); return remaining
        def reserve(self,n): self.submitted+=n; return min(600,self.remaining())
    w=support.Worker.__new__(support.Worker)
    w.contract=c; w.contract_path=Path(cp); w.run_contract_sha256=sha(cp); w.worker_sha256=sha(HERE/'entry.py')
    w.job=run; w.tele={'archive_root':bindings['archive_root']}; w.rows=rows; w.by_id={r['page_id']:r for r in rows}; w.pins=pins
    w.config=config; w.budget=Budget(); w.started=time.monotonic(); w.local=threading.local(); w.preparation_events=[]; w.preparation_lock=threading.Lock()
    w.state={'stage':'LOADING','batch':None,'calls':0,'completed_requests':0,'completed_pages':[]}
    w.bindings=[]; w.page_ids=[]; w.crop_candidates={}
    def timeout(sig,frame): raise TimeoutError('REQUEST_600S_OR_OWNER_LEASE_EXPIRED')
    signal.signal(signal.SIGALRM,timeout)
    w.install_observers(client,vllm.LLM)
    stage_raw=[]
    original_append=support.append
    def observed_append(path,value):
        original_append(path,value)
        if Path(path).name=='PRIVATE_RAW_REQUESTS.jsonl': stage_raw.append(value)
    support.append=observed_append
    adoption_ordinal=0
    for batch,pos in enumerate(range(0,len(rows),16)):
        batch_rows=rows[pos:pos+16]; d=run/'BATCHES'/f'{batch:04d}'; d.mkdir(parents=True,exist_ok=False)
        # Original stage scripts each take Path(id).stem exactly once.
        put(d/'INPUTS.json',{'bound_rows':stage_input_rows(batch_rows)})
        w.state['batch']=batch; w.page_ids=[r['page_id'] for r in batch_rows]
        object_rows={}
        def bind_object(obj,kind,row):
            object_rows[id(obj)]=Path(row['id']).stem
        def accept(obj,kind,txt):
            nonlocal adoption_ordinal
            bbox=list(obj['bbox'])
            # Search this stage's newly observed raw, never old logs or caches.
            qlist=[q for q in stage_raw if q['binding'].get('bbox')==bbox and q['binding'].get('kind')==kind]
            # Caller r is frozen stage-local row; same boxes on another page cannot bind.
            page_id=object_rows.get(id(obj))
            qlist=[q for q in qlist if q['binding'].get('page_id')==page_id]
            q=max(qlist,key=lambda q:(q['call'],q['batch_index'])) if qlist else None
            adoption_ordinal+=1
            row_identity=w.by_id.get(page_id,{})
            append(run/'ADOPTIONS.jsonl',{'ordinal':adoption_ordinal,'stage':w.state['stage'],'page_id':page_id,'input_image_sha256':row_identity.get('image_sha256'),
                'kind':kind,'bbox':bbox,'accepted_payload_sha256':hashlib.sha256(txt.encode()).hexdigest(),'call':q['call'] if q else None,
                'batch_index':q['batch_index'] if q else None,'run_contract_sha256':w.run_contract_sha256,'accepted_write_observed':True})
        try:
            for stage,args in stage_plan(d):
                if stage=='TABLE1X': materialize(d,batch_rows,'GUARD_VIEW',['GUARD','FORMULA125','NATIVE'])
                w.state['stage']=stage; w.bindings=[]; w.active_script=str(Path(bindings['archive_root'])/STAGES[stage])
                require(sha(w.active_script)==c['stage_pins'][STAGES[stage]],'STAGE_CODE_CHANGED_DURING_RUN')
                require(sha(cp)==w.run_contract_sha256,'CONTRACT_CHANGED_DURING_RUN')
                stage_raw.clear()
                if stage!='NATIVE': w.capture_crop_candidates(d,batch_rows,stage)
                append(run/'STAGES.jsonl',{'stage':stage,'batch':batch,'status':'START','utc':utc()})
                previous=sys.argv
                try:
                    sys.argv=[w.active_script,'--contract',str(d/'INPUTS.json'),'--model',bindings['model_root'],*args]
                    object_rows.clear()
                    exec(stage_code(w.active_script),{'__name__':'__main__','__file__':w.active_script,'__r3_accept__':accept,'__r3_bind__':bind_object})
                finally: sys.argv=previous
                append(run/'STAGES.jsonl',{'stage':stage,'batch':batch,'status':'COMPLETE','utc':utc()})
            materialize(d,batch_rows,'B0',['TABLE1X','GUARD_VIEW'])
            for row in batch_rows:
                append(run/'PAGES.jsonl',{'page_id':row['page_id'],'image_sha256':row['image_sha256'],'size':row['size'],'batch':batch,'status':'COMPLETE'})
        except Exception as exc:
            append(run/'ERRORS.jsonl',{'batch':batch,'stage':w.state['stage'],'error':type(exc).__name__,'message':str(exc),'pages':[r['page_id'] for r in batch_rows]})
            # No retry. Continue independent next batch; missing pages remain denominator.
            if isinstance(exc,(MemoryError,TimeoutError)) or 'out of memory' in str(exc).lower(): raise
    signal.setitimer(signal.ITIMER_REAL,0)
    verify_loaded_sources()
    import torch
    put(run/'TELE_RESULT.json',{'pages_completed':len(jsonl(run/'PAGES.jsonl')),'submitted_requests':w.budget.submitted,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'actual_gpu_phase':True,'wall_seconds':time.monotonic()-w.started})


def paddle_phase(c,cp,run):
    """Invoke exact pinned producer worker with fresh current-only manifest."""
    import runpy
    from types import SimpleNamespace
    p=c['paddle_binding']; require(sha(p['source'])==c['paddle_source_sha256'],'PADDLE_PRODUCER_PIN_CHANGED')
    require(c['resource'].get('paddle_assets_verified_manifest_sha256')==c['paddle_asset_pin_manifest_sha256'],'CURRENT_PADDLE_ASSET_IDENTITY_REQUIRED')
    output=run/'PADDLE'; output.mkdir(exist_ok=False)
    import importlib.metadata
    from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
    original_build=DocVLMLocalPredictor._build
    requests=0; phase_start=time.monotonic()
    def timeout(sig,frame): raise TimeoutError('PADDLE_REQUEST_600S_OR_OWNER_LEASE_EXPIRED')
    signal.signal(signal.SIGALRM,timeout)
    def observed_build(self,**kwargs):
        append(run/'PADDLE_MODEL_LOADS.jsonl',{'status':'START','utc':utc()})
        try:model,processor=original_build(self,**kwargs)
        except Exception as exc:
            append(run/'PADDLE_MODEL_LOADS.jsonl',{'status':'ERROR','error':type(exc).__name__,'message':str(exc),'utc':utc()});raise
        append(run/'PADDLE_MODEL_LOADS.jsonl',{'status':'COMPLETE','model_type':type(model).__name__,'processor_type':type(processor).__name__,'utc':utc()})
        generate=model.generate
        def observed_generate(*args,**options):
            nonlocal requests
            ids=options.get('input_ids'); count=int(ids.shape[0]) if ids is not None else 1
            require(options.get('max_new_tokens')==4096,'PINNED_PADDLE_OUTPUT_CAP_CHANGED')
            remaining=c['resource']['lease_expires_epoch']-time.time();require(remaining>0,'OWNER_LEASE_EXPIRED')
            requests+=count
            append(run/'PADDLE_REQUESTS.jsonl',{'requests_so_far':requests,'input_shape':list(ids.shape) if ids is not None else None,
                'actual_attention_mask_tokens':int(options['attention_mask'].sum().item()) if options.get('attention_mask') is not None else None,
                'max_new_tokens':options.get('max_new_tokens'),'processor':type(processor).__name__,'utc':utc()})
            signal.setitimer(signal.ITIMER_REAL,min(600,remaining))
            try: return generate(*args,**options)
            finally: signal.setitimer(signal.ITIMER_REAL,0)
        model.generate=observed_generate
        return model,processor
    DocVLMLocalPredictor._build=observed_build
    module=runpy.run_path(p['source'],run_name='r3_pinned_paddle_producer')
    require(callable(module.get('worker')),'PINNED_PRODUCER_WORKER_MISSING')
    module['worker'](SimpleNamespace(root=p['root'],output=str(output),manifest=str(run/'PADDLE_INPUTS.json')))
    put(run/'PADDLE_RESULT.json',{'actual_gpu_phase':True,'requests':requests,'wall_seconds':time.monotonic()-phase_start,
        'runtime_versions':{n:importlib.metadata.version(n) for n in ['paddleocr','paddlex','paddlepaddle-gpu']},'source_sha256':sha(p['source'])})


def fresh_page(run,row):
    index=row['local_index']//16; d=owned(run,run/'BATCHES'/f'{index:04d}')
    # A partial stage can have a markdown file but an unfinished middle write;
    # only a completed four-stage B0 pair is accepted by this first entry.
    md=owned(run,d/'B0/markdown'/(row['page_id']+'.md'))
    middle=owned(run,d/'B0/middle'/(row['page_id']+'.json'))
    if not md.is_file() or not middle.is_file(): return None,None
    return md.read_bytes().decode('utf-8'),load(middle)


def supervisor(c,cp,rows,run,driver=None):
    """driver injection is for explicitly synthetic CPU controls only."""
    from current_r3 import compose
    from paddle_contract import validate_donor
    start=time.monotonic(); started_epoch=time.time();phases={}; synthetic=driver is not None; contract_sha=sha(cp)
    row_list=[dict(r,page_id=Path(r['id']).stem,local_index=i) for i,r in enumerate(rows)]
    if driver is None:
        def driver(phase):
            phase_start=time.monotonic()
            python=c['bindings']['tele_python'] if phase=='tele' else c['paddle_binding']['python']
            require(sha(cp)==contract_sha,'CONTRACT_CHANGED_DURING_RUN')
            from device_compat import numeric_child_env
            env=dict(numeric_child_env(c['resource'],os.environ),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',R3_PARENT_CHILD='1',R3_CONTRACT_SHA256=contract_sha)
            with (run/(phase+'.log')).open('xb') as log:
                remaining=c['resource']['lease_expires_epoch']-time.time()
                require(remaining>0,'OWNER_LEASE_EXPIRED')
                child=subprocess.Popen([python,str(HERE/'entry.py'),phase,'--contract',str(cp)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    # Only this supervisor's newly created process group, no scan/other owner.
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()
                    return {'returncode':child.returncode,'error':'OWNER_LEASE_EXPIRED','wall_seconds':time.monotonic()-phase_start}
            return {'returncode':code,'wall_seconds':time.monotonic()-phase_start}
    try: phases['tele']=driver('tele')
    except Exception as exc: phases['tele']={'error':type(exc).__name__,'message':str(exc)}
    records=jsonl(run/'PRIVATE_RAW_REQUESTS.jsonl'); adoptions=jsonl(run/'ADOPTIONS.jsonl'); pages={}; eligible=[]; errors={}
    stage_pins={s:c['stage_pins'][n] for s,n in STAGES.items()}
    for r in row_list:
        try:
            md,middle=fresh_page(run,r); pages[r['page_id']]=(md,middle)
            if md is None: continue
            _,_,trigger=compose(r,md,middle,records,adoptions,contract_sha,stage_pins,selection_policy=c.get('selection_policy','historical_latest_abnormal'))
            if trigger: eligible.append({'page_id':r['page_id'],'image_path':str(Path(c['bindings']['image_root'])/r['id']),'input_sha256':r['image_sha256']})
        except Exception as exc: errors[r['page_id']]=str(exc)
    put(run/'PADDLE_INPUTS.json',{'pages':eligible,'reference_access':False,'run_contract_sha256':contract_sha})
    if eligible:
        try:
            require(all(c['paddle_binding'].get(k) for k in ['source','python','root']),'PADDLE_BINDING_MISSING')
            phases['paddle']=driver('paddle')
        except Exception as exc: phases['paddle']={'error':type(exc).__name__,'message':str(exc)}
    else: phases['paddle']={'status':'NOT_TRIGGERED','actual_path_coverage':False}
    terminals=[]
    for r in row_list:
        md,middle=pages.get(r['page_id'],(None,None)); decisions=[]; final=md or ''; state='COMPLETE' if md is not None else 'MISSING'
        if md is not None:
            donor,donor_md=None,None; folder=owned(run,run/'PADDLE'/r['page_id'])
            try:
                if (folder/'native.json').is_file():
                    d=load(folder/'native.json'); donor=d.get('res',d)
                    validate_donor(donor,middle['pdf_info'][0]['page_size'])
                    donor_md=(folder/'prediction.md').read_text(encoding='utf-8')
                final,decisions,_=compose(r,md,middle,records,adoptions,contract_sha,stage_pins,donor,donor_md,selection_policy=c.get('selection_policy','historical_latest_abnormal'))
                if any(e['page_id']==r['page_id'] for e in eligible) and donor is None:
                    state='ERROR_RETAINED'
            except Exception as exc:
                final=md; decisions=[{'state':'UNKNOWN_DONOR_OR_CURRENT_STRUCTURE','action':'retain','error':str(exc)}]; state='ERROR_RETAINED'
        dest=run/'R3/markdown'/(r['page_id']+'.md'); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(final.encode('utf-8'))
        terminals.append({'id':r['id'],'state':state,'in_denominator':True,'prediction_sha256':sha(dest),'decisions':decisions,'error':errors.get(r['page_id'])})
    put(run/'TERMINALS.json',terminals)
    phase_failed=any('error' in p or p.get('returncode',0)!=0 for p in phases.values())
    result={'status':'CPU_SYNTHETIC_ONLY' if synthetic else 'FINISHED_WITH_ERRORS' if any(t['state']!='COMPLETE' for t in terminals) or phase_failed else 'FINISHED',
        'candidate':c['candidate'],'selection_policy':c.get('selection_policy','historical_latest_abnormal'),'pages':len(terminals),'returned_pages':sum(t['state']=='COMPLETE' for t in terminals),
        'paddle_trigger_pages':len(eligible),'phases':phases,'GPU_smoke':'NOT_RUN' if synthetic else 'REQUIRES_RESULT_REVIEW',
        'old_cache_inputs':[],'score':None,'contract_sha256':contract_sha,'synthetic':synthetic}
    if c.get('experiment_kind')=='FRESH_FORMAL_RUN':
        from formal_terminal import summarize
        terminal=summarize(rows,terminals,jsonl(run/'STAGES.jsonl'),jsonl(run/'PAGES.jsonl'),jsonl(run/'SOURCE_AUDITS.jsonl'),phase_failed,synthetic)
        result.update(experiment_kind='FRESH_FORMAL_RUN',formal_terminal=terminal,source_manifest_sha256=c['source_manifest_sha256'],
            inputs_sha256=c['inputs_sha256'],scope=c['scope'],terminal_roster_sha256=sha(run/'TERMINALS.json'),
            continuation_is_cold_start_pass=False,cold_start_smoke_pass=False,
            formal_acceptance_evidence=c['formal_acceptance_evidence'],reused_source_requests=0)
        if not synthetic and not terminal['protocol_completed']:result['status']='FINISHED_WITH_ERRORS'
    cost={'wall_seconds':time.monotonic()-start,'GPU_card_seconds':0 if synthetic else None,
        'single_GPU_reserved_envelope_seconds':0 if synthetic else time.monotonic()-start,
        'GPU_cost_status':'NOT_RUN' if synthetic else 'phase_wall_includes_CPU_and_model_load; allocation cost requires parent telemetry',
        'attempts':phases,'no_retries':True}
    if c.get('experiment_kind')=='FRESH_FORMAL_RUN':
        from fresh_cost import cost as account
        cost=account(records,jsonl(run/'PADDLE_REQUESTS.jsonl'),phases,time.monotonic()-start,synthetic,
            loaded=load(run/'LOADED.json') if (run/'LOADED.json').is_file() else None,
            tele_result=load(run/'TELE_RESULT.json') if (run/'TELE_RESULT.json').is_file() else None,
            paddle_loads=jsonl(run/'PADDLE_MODEL_LOADS.jsonl'),started_epoch=started_epoch,ended_epoch=time.time())
    put(run/'COST.json',cost)
    if c.get('experiment_kind')=='FRESH_FORMAL_RUN':result['cost_sha256']=sha(run/'COST.json')
    put(run/'RESULT.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['check','run','tele','paddle']); ap.add_argument('--contract',required=True,type=Path); a=ap.parse_args()
    cp=a.contract.resolve(); c=load(cp)
    if a.mode=='check':
        rows=validate(c,cp); print(json.dumps({'status':'CPU_CONTRACT_PASS','pages':len(rows),'GPU_smoke':'NOT_RUN','bindings_complete':all(c['bindings'].values())})); return
    if a.mode=='run':
        rows=validate(c,cp,live=True); run=Path(c['bindings']['output_root']); run.mkdir(parents=True,exist_ok=False)
        # Parent has admission responsibility; check selected physical card identity again.
        snapshot=subprocess.run(['nvidia-smi','-i',str(c['resource']['gpu_index']),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
        fields=[s.strip() for s in snapshot.stdout.strip().split(',')]
        require(len(fields)==3 and fields[0]==c['resource']['gpu_uuid'] and int(fields[1])==0 and int(fields[2])==0,'GPU_NOT_FRESH_IDLE_EXACT_UUID')
        result=supervisor(c,cp,rows,run); print(json.dumps(result))
        if c.get('experiment_kind')=='FRESH_FORMAL_RUN' and not result['formal_terminal']['protocol_completed']:raise SystemExit(2)
        return
    require(os.environ.get('R3_PARENT_CHILD')=='1','REGISTERED_PARENT_SUPERVISOR_REQUIRED')
    require(sha(cp)==os.environ.get('R3_CONTRACT_SHA256'),'CHILD_CONTRACT_CHANGED')
    require(time.time()<c['resource']['lease_expires_epoch'],'OWNER_LEASE_EXPIRED')
    run=Path(c['bindings']['output_root'])
    try:
        from device_compat import verify_live
        put(run/(a.mode.upper()+'_DEVICE_MAPPING.json'),verify_live(c['resource']))
        if a.mode=='tele': tele_phase(c,cp,run)
        else: paddle_phase(c,cp,run)
    except BaseException as exc:
        put(run/(a.mode.upper()+'_ERROR.json'),{'error':type(exc).__name__,'message':str(exc),'utc':utc()}); traceback.print_exc(); raise


if __name__=='__main__': main()
