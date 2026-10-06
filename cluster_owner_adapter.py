"""Owner-side bridge. Public CLI cannot obtain an admission context or submit jobs.

The cluster owner supplies the existing fresh identity/health/lease/lock admission
context. No historical admission file or hardcoded host identity is accepted here.
The owner's original adapter and its safeguards must remain in use.
"""
import time
from pathlib import Path
from assets import sha, write
from inference import checked_rows, pipeline, verify_code


def execute_admitted(contract, contract_path, admission_context, probe=False, max_pages=1, max_generation_items=32):
    if contract.get('execution_policy') != 'cluster':
        raise ValueError('Explicit cluster policy required.')
    rows = checked_rows(contract)
    if type(max_pages) is not int or not 1 <= max_pages <= 20 or len(rows) > max_pages:
        raise ValueError('Explicit bounded owner scope exceeded.')
    if type(max_generation_items) is not int or not 1 <= max_generation_items <= 32:
        raise ValueError('Explicit owner generation scope must be 1..32.')
    verify_code(contract)
    # Only the owner can construct this already-authorized context; this bridge
    # does not set parent_admitted, renew leases, acquire identities or register.
    with admission_context as resource:
        if resource.get('parent_admitted') is not True or resource.get('owner_lease_verified') is not True:
            raise ValueError('Fresh parent/owner admission required.')
        index, identity = resource.get('gpu_index'), resource.get('gpu_uuid')
        if type(index) is not int or not 1 <= index <= 6 or not isinstance(identity, str) or not identity.startswith('GPU-'):
            raise ValueError('Admitted non-GPU0/7 single-device identity required.')
        if index != contract['runtime']['device_index']:
            raise ValueError('Prepared device differs from owner admission.')
        remaining = resource.get('lease_expires_epoch', 0) - time.time()
        if remaining <= 0:
            raise ValueError('Owner lease expired; no borrowing or renewal.')
        contract['runtime']['device_id'] = identity
        # These are fresh non-secret owner metadata, never invented by this
        # bridge. Original device_compat.verify_live verifies them in each child.
        if type(resource.get('uid')) is not int or not isinstance(resource.get('boot_id'), str):
            raise ValueError('Actual fresh owner UID/boot identity required.')
        contract['cluster_resource'] = dict(resource)
        contract['runtime']['owner_admission_expires_epoch'] = resource['lease_expires_epoch']
        contract['phase_timeout_seconds'] = min(contract['phase_timeout_seconds'], int(remaining))
        contract['max_generation_items'] = max_generation_items
        from generation_budget import GenerationBudget
        GenerationBudget(contract['output'], max_generation_items).initialize()
        write(contract_path, contract)
        result = pipeline(contract, Path(contract_path), probe=probe)
        result['owner_admission_used'] = True
        result['contract_sha256'] = sha(contract_path)
        write(Path(contract['output']) / 'RESULT.json', result)
        return result
