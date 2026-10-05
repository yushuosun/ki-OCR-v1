"""Fail-closed numeric visibility for vLLM 0.11; no CUDA context allocation.

Only the registered supervisor creates this environment. The external runner
continues binding by UUID. Physical ordinal is never inferred from logical 0.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def numeric_child_env(resource, parent_env):
    index = resource['gpu_index']
    require(type(index) is int and 1 <= index <= 6, 'GPU_INDEX_FORBIDDEN')
    require(parent_env.get('CUDA_VISIBLE_DEVICES') == resource['gpu_uuid'],
            'REGISTERED_OUTER_UUID_MISMATCH')
    return dict(parent_env, CUDA_VISIBLE_DEVICES=str(index),
                CUDA_DEVICE_ORDER='PCI_BUS_ID',
                R3_ADMITTED_GPU_UUID=resource['gpu_uuid'])


def check_identity(resource, env, actual_uid, actual_boot, now, smi_uuid):
    index = resource['gpu_index']
    require(type(index) is int and 1 <= index <= 6, 'GPU_INDEX_FORBIDDEN')
    require(env.get('CUDA_VISIBLE_DEVICES') == str(index), 'NUMERIC_VISIBILITY_MISMATCH')
    require(env.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'DEVICE_ORDER_MISMATCH')
    require(env.get('R3_ADMITTED_GPU_UUID') == resource['gpu_uuid'], 'ADMITTED_UUID_MISMATCH')
    require(actual_uid == resource['uid'] and actual_boot == resource['boot_id'], 'BOOT_UID_CHANGED')
    require(now < resource['lease_expires_epoch'], 'OWNER_LEASE_EXPIRED')
    require(smi_uuid == resource['gpu_uuid'], 'PHYSICAL_INDEX_UUID_CHANGED')


def check_cuda_mapping(resource, count, logical_uuid):
    require(count == 1, 'CUDA_SINGLE_DEVICE_REQUIRED')
    require(logical_uuid == resource['gpu_uuid'], 'CUDA_ORDINAL_UUID_MISMATCH')


def driver_identity():
    # Called exclusively by the fixed short-lived exec child below. Never in Tele.
    import ctypes
    lib = ctypes.CDLL('libcuda.so.1')
    def call(name, *args):
        result = getattr(lib, name)(*args)
        require(result == 0, 'CUDA_DRIVER_API_FAILED:%s:%s' % (name, result))
    call('cuInit', ctypes.c_uint(0))
    count = ctypes.c_int()
    call('cuDeviceGetCount', ctypes.byref(count))
    require(count.value == 1, 'CUDA_SINGLE_DEVICE_REQUIRED')
    device = ctypes.c_int()
    call('cuDeviceGet', ctypes.byref(device), ctypes.c_int(0))
    raw = (ctypes.c_ubyte * 16)()
    # Base UUID is also what nvidia-smi reports for the non-MIG admitted cards.
    call('cuDeviceGetUuid', ctypes.byref(raw), device)
    return count.value, 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


def isolated_driver_identity(resource):
    remaining = resource['lease_expires_epoch'] - time.time()
    require(remaining > 0, 'OWNER_LEASE_EXPIRED')
    # Fixed executable + fixed checked-in file/mode, no shell or caller program.
    child = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                            '--driver-identity'], env=dict(os.environ),
                           check=True, capture_output=True, text=True,
                           timeout=min(15, remaining))
    data = json.loads(child.stdout)
    require(isinstance(data, dict) and set(data) == {'count', 'uuid'}, 'DRIVER_RESULT_SCHEMA')
    require(type(data['count']) is int and isinstance(data['uuid'], str), 'DRIVER_RESULT_TYPES')
    return data['count'], data['uuid']


def verify_live(resource):
    smi = subprocess.run(['nvidia-smi', '-i', str(resource['gpu_index']),
                          '--query-gpu=uuid', '--format=csv,noheader'],
                         check=True, capture_output=True, text=True, timeout=15)
    check_identity(resource, os.environ, os.getuid(),
                   Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   time.time(), smi.stdout.strip())
    count, actual_uuid = isolated_driver_identity(resource)
    require(time.time() < resource['lease_expires_epoch'], 'OWNER_LEASE_EXPIRED')
    check_cuda_mapping(resource, count, actual_uuid)
    return {'physical_index': resource['gpu_index'], 'expected_uuid': resource['gpu_uuid'],
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'cuda_device_order': os.environ['CUDA_DEVICE_ORDER'],
            'logical_device_count': count, 'logical_device_0_uuid': actual_uuid,
            'boot_id': resource['boot_id'], 'uid': resource['uid'],
            'lease_expires_epoch': resource['lease_expires_epoch'],
            'driver_query_isolation': 'fixed_short_lived_exec_child; no cuInit in model parent',
            'gpu_model_smoke': 'NOT_PROVED_BY_DRIVER_METADATA'}


if __name__ == '__main__':
    require(sys.argv[1:] == ['--driver-identity'], 'FIXED_DRIVER_MODE_ONLY')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible in {'1', '2', '3', '4', '5', '6'}, 'GPU_INDEX_FORBIDDEN')
    require(os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'DEVICE_ORDER_MISMATCH')
    require(os.environ.get('R3_ADMITTED_GPU_UUID', '').startswith('GPU-'), 'ADMITTED_UUID_REQUIRED')
    count, actual_uuid = driver_identity()
    print(json.dumps({'count': count, 'uuid': actual_uuid}))
