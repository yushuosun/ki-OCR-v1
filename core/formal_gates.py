"""Read actual pinned acceptance evidence; no submit/install or invented PASS."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path('/work/docai/hybrid_v31_cluster_smoke_20261001_r1')
MANIFEST='363cdb89f7ba01e742474454011e36e57d9601566baf4660ce94ef52d7b711c8'
INPUTS={'gpu2':'81727097cd6019904f4796ad75e7b88c7aa49bc37a6d98345a2a423ac2c181bf',
        'gpu4':'187d0bf3704dd17ae2b46079de39fed3ff3eb8427a3df94a2b619665b6e609f3'}
FORBIDDEN=('source_run_id','source_contract_sha256','resume_native','continuation_source','reused_native_requests')


def require(ok,reason):
    if not ok:raise ValueError(reason)


def fresh_contract(c):
    require(c.get('experiment_kind')=='FRESH_FORMAL_RUN','FRESH_FORMAL_KIND_REQUIRED')
    require(not any(key in c for key in FORBIDDEN),'FRESH_FORMAL_NO_SOURCE_RESUME')
    require(c['old_cache_inputs']==[],'FRESH_FORMAL_OLD_CACHE_FORBIDDEN')
    require(c['scope'] in INPUTS,'FIXED_FORMAL_SLICE_REQUIRED')
    require(c['inputs_sha256']==INPUTS[c['scope']],'FROZEN_FULL_INPUT_SHA_REQUIRED')
    require(c.get('smoke_pass_contract_sha256') is None,'DO_NOT_INVENT_COLD_SMOKE_PASS')
    require(c.get('formal_acceptance_evidence',{}).get('continuation_is_cold_start_pass') is False,'CONTINUATION_NOT_COLD_START_PASS')


def read_evidence(ref,root):
    require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and
            isinstance(ref.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'MISSING_ACTUAL_PINNED_EVIDENCE')
    path=Path(ref['path']);root=Path(root).resolve()
    require(path.is_absolute() and path.resolve().is_relative_to(root) and
            path.resolve()==Path(os.path.abspath(path)) and not path.is_symlink(),'EVIDENCE_OUTSIDE_PROJECT_OR_LINK')
    require(path.is_file() and path.stat().st_size<=2097152,'BOUNDED_EVIDENCE_REQUIRED')
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==ref['sha256'],'ACTUAL_EVIDENCE_SHA_CHANGED')
    value=json.loads(data);require(isinstance(value,dict),'EVIDENCE_OBJECT_REQUIRED')
    return value


def acceptance(c,root=ROOT):
    fresh_contract(c);gate=c['formal_acceptance_evidence']
    require(gate.get('user_explicit_formal_approval_confirmed') is True and
            isinstance(gate.get('user_explicit_formal_approval'),str) and gate['user_explicit_formal_approval'].strip(), 'EXPLICIT_PARENT_USER_FORMAL_APPROVAL_REQUIRED')
    result=read_evidence(gate.get('continuation_result'),root)
    contract=read_evidence(gate.get('continuation_contract'),root)
    require(result.get('synthetic') is False and result.get('status')=='FINISHED' and
            result.get('experiment_kind')=='SOURCE_RUN_CONTINUATION' and result.get('cold_start_smoke_pass') is False and
            result.get('pages')==1 and result.get('returned_pages')==1 and result.get('reused_native_requests')==16 and
            type(result.get('new_tele_records')) is int and result['new_tele_records']>=1 and
            result.get('logical_source_run_id')=='run_81e0dc870591a55a95fd325f6f905d42','REAL_SUCCESSFUL_CONTINUATION_REQUIRED')
    require(result.get('contract_sha256')==gate['continuation_contract']['sha256'] and
            contract.get('experiment_kind')=='SOURCE_RUN_CONTINUATION' and
            contract.get('resource',{}).get('gpu_index')==2 and
            bool(re.fullmatch('run_[0-9a-f]{32}',contract.get('resource',{}).get('backend_run_id',''))), 'CONTINUATION_ACTUAL_CONTRACT_BINDING_REQUIRED')
    require(all('error' not in phase and phase.get('returncode',0)==0 for phase in result.get('phases',{}).values()) and
            result.get('phases',{}).get('tele',{}).get('returncode')==0,'CONTINUATION_PHASES_MUST_PASS')
    staging=read_evidence(gate.get('image_staging_result'),root)
    require(staging.get('state')=='BOTH_FORMAL_IMAGE_ROSTERS_VERIFIED' and staging.get('roster_pages')==1651 and
            staging.get('source_original_manifest_sha256')==MANIFEST,'ACTUAL_FULL1651_STAGING_REQUIRED')
    require([(h['half'],h['roster_pages']) for h in staging.get('halves',[])]==[('front826',826),('back825',825)],'BOTH_STAGING_ROSTERS_REQUIRED')
    staging_accept=staging.get('accepted_continuation',{})
    require(staging_accept.get('result_sha256')==gate['continuation_result']['sha256'] and
            staging_accept.get('contract_sha256')==gate['continuation_contract']['sha256'] and
            staging_accept.get('cold_start_smoke_pass') is False,'STAGING_CONTINUATION_BINDING_REQUIRED')
    review=read_evidence(gate.get('independent_formal_review'),root)
    require(review.get('status')=='PASS' and review.get('review_kind')=='INDEPENDENT_FORMAL_REVIEW' and
            isinstance(review.get('reviewer_id'),str) and review['reviewer_id'] and
            review.get('reviewer_is_implementation_owner') is False,'INDEPENDENT_FORMAL_REVIEW_REQUIRED')
    require(review.get('entry_pins')==c['entry_pins'] and review.get('original_manifest_sha256')==MANIFEST and
            review.get('continuation_result_sha256')==gate['continuation_result']['sha256'] and
            review.get('continuation_accepted') is True,'REVIEW_MUST_BIND_FINAL_CODE_AND_ACTUAL_CONTINUATION')
    resource=read_evidence(gate.get('resource_contract'),root)
    r=c['resource']
    require(resource.get('status')=='APPROVED_FORMAL_RESOURCE_CONTRACT' and resource.get('scope')==c['scope'] and
            resource.get('gpu_index')==c['fixed_gpu_index']==r['gpu_index'] and
            resource.get('uid')==r['uid'] and resource.get('boot_id')==r['boot_id'] and
            resource.get('gpu_uuid')==r['gpu_uuid'],'ACTUAL_FORMAL_RESOURCE_CONTRACT_REQUIRED')
    seconds=resource.get('approved_formal_lease_seconds');start=r.get('lease_started_epoch');end=r.get('lease_expires_epoch')
    require(type(seconds) is int and seconds>0 and type(start) in (int,float) and type(end) in (int,float) and
            0<end-start<=seconds,'APPROVED_FORMAL_LEASE_REQUIRED_NO_SMOKE_DEFAULT')
    return {'status':'ACTUAL_FORMAL_ACCEPTANCE_PASS','continuation_is_cold_start_pass':False,
            'reviewed_continuation_result_sha256':gate['continuation_result']['sha256'],
            'evidence_sha256':{name:gate[name]['sha256'] for name in ('continuation_result','continuation_contract','image_staging_result','independent_formal_review','resource_contract')}}
