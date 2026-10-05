"""R3 on this run only; historical policy functions imported unchanged.

Raw records must carry this contract, page, input and original stage identity.
Default reproduction selects latest abnormal raw by historical stage/UTC order.
AST accepted-write ownership is a separately named, opt-in candidate policy.
"""
from collections import defaultdict
from cascade import box, payload_type, effective_state, recover_then_fallback, unique_overlap, exact_patch
from native import convert

HISTORICAL = 'historical_latest_abnormal'
AST_OWNER = 'ast_accepted_owner'
POLICY_CANDIDATES = {HISTORICAL:'R3_VLLM_COLD_SAME_BACKEND_R1', AST_OWNER:'R3_VLLM_COLD_AST_OWNER_CANDIDATE_R1'}
RANK = {'NATIVE':0,'FORMULA125':1,'GUARD':2,'TABLE1X':3}


def abnormal(q):
    return ('outputs' in q and len(q['outputs']) != 1) or not isinstance(q.get('raw_text'),str) or not q['raw_text'].strip() or q.get('finish_reason') != 'stop'


def slots(m):
    page = m['pdf_info'][0]
    result = []
    def walk(blocks, path):
        for i, b in enumerate(blocks):
            for j, line in enumerate(b.get('lines', [])):
                for k, sp in enumerate(line.get('spans', [])):
                    kind = payload_type(sp.get('type'))
                    if kind != 'image':
                        result.append({'path': path+[i,'lines',j,'spans',k], 'kind': kind,
                            'bbox': sp.get('bbox'), 'box': box(sp.get('bbox'), page['page_size']),
                            'payload': sp.get('html' if kind == 'table' else 'content')})
            walk(b.get('blocks', []), path+[i,'blocks'])
    walk(page.get('para_blocks', []), ['pdf_info',0,'para_blocks'])
    walk(page.get('discarded_blocks', []), ['pdf_info',0,'discarded_blocks'])
    return result


def events_for(row, records, contract_sha, stage_pins):
    result = []
    for q in records:
        b, source = q.get('binding', {}), q.get('source', {})
        if not isinstance(b, dict) or not isinstance(source, dict):
            continue
        if q.get('run_contract_sha256') != contract_sha or b.get('page_id') != row['page_id'] or b.get('input_image_sha256') != row['image_sha256']:
            continue
        if source.get('script_sha256') != stage_pins.get(q.get('stage')):
            continue
        if not isinstance(b.get('bbox'), list) or len(b['bbox']) != 4 or b.get('kind') in (None,'layout','UNKNOWN'):
            continue
        if 'outputs' in q and len(q['outputs']) != 1:
            q=dict(q,raw_text=None,finish_reason=None)  # historical cardinality UNKNOWN may outrank a failure
        result.append(q)
    return result


def compose(row, md, middle, records, adoptions, contract_sha, stage_pins, donor=None, donor_md=None, selection_policy=HISTORICAL):
    if selection_policy not in POLICY_CANDIDATES: raise ValueError('UNKNOWN_R3_SELECTION_POLICY')
    if not md.strip():
        if donor and [donor.get('width'), donor.get('height')] == middle['pdf_info'][0]['page_size'] and isinstance(donor_md,str) and donor_md.strip():
            return donor_md, [{'state':'CURRENT_WHITESPACE_EMPTY_PAGE','action':'PADDLE_WHOLE_PAGE_FALLBACK'}], True
        return md, [{'state':'CURRENT_WHITESPACE_EMPTY_PAGE','action':'UNKNOWN_WHOLE_PAGE_BACKUP'}], True
    current = slots(middle)
    events = events_for(row, records, contract_sha, stage_pins)
    by_slot = defaultdict(list)
    for q in events:
        b = q['binding']
        by_slot[(payload_type(b['kind']), tuple(b['bbox']))].append(q)
    accepted = defaultdict(list)
    for a in adoptions:
        if a.get('run_contract_sha256') == contract_sha and a.get('page_id') == row['page_id'] and a.get('input_image_sha256') == row['image_sha256']:
            accepted[(a['kind'], tuple(a['bbox']))].append(a)
    backups = []
    if donor is not None:
        for i,b in enumerate(donor['parsing_res_list']):
            if b['block_label'] not in {'image','header_image','footer_image'}:
                # Preserve ORIGINAL R3 payload_type, including its formula-label limitation.
                backups.append({'kind':payload_type(b['block_label']), 'box':box(b['block_bbox'],[donor['width'],donor['height']]), 'payload':b['block_content'],'index':i})
    final, decisions, eligible = md, [], False
    targets=current
    if selection_policy==HISTORICAL:
        # Match the original events dict's insertion order: abnormal raw order,
        # then empty slots without an abnormal event. No AST ownership filtering.
        keys=[]
        for q in events:
            b=q['binding']; key=(payload_type(b['kind']),tuple(b['bbox']))
            if abnormal(q) and key not in keys: keys.append(key)
        for s in current:
            key=(s['kind'],tuple(s['bbox'] or []))
            if (not isinstance(s['payload'],str) or not s['payload'].strip()) and key not in keys: keys.append(key)
        targets=[s for key in keys for s in current if (s['kind'],tuple(s['bbox'] or []))==key]
    for target in targets:
        key = (target['kind'], tuple(target['bbox'] or []))
        if sum(s['kind']==target['kind'] and s['bbox']==target['bbox'] for s in current) != 1:
            decisions.append({'state':'UNKNOWN_SLOT_GEOMETRY_AMBIGUOUS','action':'retain'}); continue
        if selection_policy==HISTORICAL:
            choices=[q for q in by_slot[key] if abnormal(q)]
            if choices: choices=[max(choices,key=lambda q:(RANK.get(q['stage'],-1),q.get('utc') or ''))]
            elif not isinstance(target['payload'],str) or not target['payload'].strip():
                choices=[q for q in by_slot[key] if q['stage']=='NATIVE']
        else:
            owner = max(accepted[key], key=lambda a:a['ordinal']) if accepted[key] else None
            if owner:
                choices = [q for q in by_slot[key] if q.get('call') == owner['call'] and q.get('batch_index') == owner['batch_index'] and q.get('stage') == owner['stage']]
            else: choices = [q for q in by_slot[key] if q['stage']=='NATIVE']
        if len(choices) != 1:
            if by_slot[key] or not (isinstance(target['payload'],str) and target['payload'].strip()):
                decisions.append({'state':'UNKNOWN_CURRENT_RAW_BINDING','action':'retain','bbox':target['bbox']})
            continue
        q = choices[0]
        converter = lambda raw: convert(raw,target['kind'])
        state = effective_state(target['payload'],q,converter)
        d = {'state':state,'action':'retain','bbox':target['bbox'],'kind':target['kind'],'stage':q['stage'],'finish_reason':q.get('finish_reason'),'call':q['call'],'selection_policy':selection_policy}
        if state in {'EXPLICIT_FAILED_SLOT','EMPTY_RETURNED_SLOT','DETERMINISTIC_RECOVERY_FIRST'}:
            backup, geometry = unique_overlap(target,current,backups)
            replacement, action = recover_then_fallback(target['payload'],state,q,converter,None if backup is None else backup['payload'])
            # Deterministic native recovery precedes donor generation. A failed
            # exact anchor cannot be repaired by spending a Paddle request.
            eligible = eligible or action != 'DETERMINISTIC_RECOVERED'
            d.update(action=action,geometry=geometry)
            if action in {'PADDLE_FIXED_FALLBACK','DETERMINISTIC_RECOVERED'}:
                candidate, patch = exact_patch(final,target['payload'],replacement)
                d['patch_state'] = patch
                if patch=='PATCHED_EXACT_INTERVAL': final=candidate
                else: d['action']='retain_'+patch
        decisions.append(d)
    return final, decisions, eligible
