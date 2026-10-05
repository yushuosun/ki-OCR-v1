"""Read-only final 826/825 terminal aggregation; no inference, GT or scorer."""
import argparse
import hashlib
import json
from pathlib import Path
from formal_gates import MANIFEST,INPUTS,require
INPUT_ROOT=Path(__file__).resolve().parent
INPUT_FILES={'gpu2':'GPU2_INPUTS.json','gpu4':'GPU4_INPUTS.json'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_half(path,scope,count):
    path=Path(path);result=json.loads(path.read_bytes());terminal_path=path.parent/'TERMINALS.json'
    input_path=INPUT_ROOT/INPUT_FILES[scope]
    require(sha(input_path)==INPUTS[scope],'ACTUAL_FIXED_INPUT_SHA_REQUIRED')
    frozen=json.loads(input_path.read_bytes());rows=frozen['bound_rows']
    start=0 if scope=='gpu2' else 826
    require(frozen.get('source_manifest_sha256')==MANIFEST and len(rows)==count and
            [r['original_roster_index'] for r in rows]==list(range(start,start+count)) and
            len({r['id'] for r in rows})==count,'ACTUAL_FIXED_INPUT_ROSTER_REQUIRED')
    require(result.get('synthetic') is False and result.get('experiment_kind')=='FRESH_FORMAL_RUN','REAL_FRESH_FORMAL_RESULT_REQUIRED')
    require(result.get('scope')==scope and result.get('pages')==count and result.get('source_manifest_sha256')==MANIFEST and
            result.get('inputs_sha256')==INPUTS[scope] and result.get('reused_source_requests')==0,'FROZEN_HALF_IDENTITY_REQUIRED')
    require(sha(terminal_path)==result.get('terminal_roster_sha256'),'ACTUAL_TERMINAL_SHA_CHANGED')
    terminals=json.loads(terminal_path.read_bytes())
    require([t['id'] for t in terminals]==[r['id'] for r in rows],'TERMINAL_IDS_MUST_MATCH_FIXED_INPUTS_IN_ORDER')
    require(len(terminals)==count and len({t['id'] for t in terminals})==count and
            all(t.get('in_denominator') is True for t in terminals),'ORIGINAL_HALF_DENOMINATOR_REQUIRED')
    complete=sum(t.get('state')=='COMPLETE' for t in terminals)
    require(complete==result.get('returned_pages') and result.get('formal_terminal',{}).get('terminal_roster_complete') is True,
            'ACTUAL_TERMINAL_COUNT_MISMATCH')
    return result,terminals,sha(path)


def merge(front,back):
    a,ta,sha_a=read_half(front,'gpu2',826);b,tb,sha_b=read_half(back,'gpu4',825)
    require(not ({t['id'] for t in ta}&{t['id'] for t in tb}),'HALF_OVERLAP_FORBIDDEN')
    terminals=ta+tb;complete=sum(t.get('state')=='COMPLETE' for t in terminals)
    passed=all(r.get('status')=='FINISHED' and r.get('formal_terminal',{}).get('protocol_completed') is True for r in (a,b))
    return {'experiment_kind':'FRESH_FORMAL_RUN_1651_AGGREGATE','status':'FULL1651_PROTOCOL_COMPLETE' if passed else 'FULL1651_TERMINALS_WITH_FAILURES_RETAINED',
            'original_denominator_pages':1651,'terminal_pages':len(terminals),'complete_pages':complete,'failed_pages_retained':1651-complete,
            'overlap':0,'protocol_completed':passed,'cold_start_smoke_pass':False,'score':None,'source_manifest_sha256':MANIFEST,
            'half_result_sha256':[sha_a,sha_b],'GT_modified':False,'failed_page_ids':[t['id'] for t in terminals if t.get('state')!='COMPLETE']}


def merge_cost(front,back):
    halves=[]
    for result_path in (Path(front),Path(back)):
        result=json.loads(result_path.read_bytes());cp=result_path.parent/'COST.json'
        require(sha(cp)==result.get('cost_sha256'),'ACTUAL_COST_SHA_REQUIRED')
        cost=json.loads(cp.read_bytes());require(cost.get('accounting_scope')=='FRESH_FORMAL_RUN_THIS_RUN_ONLY' and
            cost.get('synthetic') is False and cost.get('source',{}).get('requests')==0,'REAL_FRESH_COST_ONLY')
        halves.append(cost)
    def add(values):return sum(values) if all(type(v) in (int,float) for v in values) else None
    totals={key:add([h['combined'].get(key) for h in halves]) for key in ('requests','actual_engine_prompt_tokens','actual_output_tokens','paddle_requests','GPU_card_seconds')}
    starts=[h['incremental'].get('supervisor_started_epoch') for h in halves];ends=[h['incremental'].get('supervisor_ended_epoch') for h in halves]
    makespan=max(ends)-min(starts) if all(type(v) in (int,float) for v in starts+ends) else None
    return {'accounting_scope':'FULL1651_TWO_FRESH_HALVES','source_requests':0,'combined':totals,
            'GPU_card_seconds':totals['GPU_card_seconds'],'parallel_supervisor_envelope_seconds':makespan,
            'sum_half_wall_seconds':add([h['incremental'].get('wall_seconds') for h in halves]),
            'model_loads_and_failed_attempts_by_half':[h['incremental'] for h in halves],
            'note':'GPU occupancy requires owner telemetry; parallel supervisor envelope is not GPU card-seconds'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--front',required=True,type=Path);parser.add_argument('--back',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--cost-out',type=Path)
    args=parser.parse_args();result=merge(args.front,args.back)
    with args.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    if args.cost_out:
        with args.cost_out.open('x',encoding='utf8') as f:json.dump(merge_cost(args.front,args.back),f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
