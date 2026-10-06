"""CPU negative controls for the owner bridge; all identities are synthetic."""
import contextlib
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cluster_owner_adapter as adapter
spec = importlib.util.spec_from_file_location('frozen_device_compat',
    Path(__file__).resolve().parents[1] / 'core/device_compat.py')
device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device)


class AdmissionControls(unittest.TestCase):
    def contract(self):
        return {'execution_policy': 'cluster', 'runtime': {'device_index': 2}}

    def reject(self, contract, resource, message):
        with patch.object(adapter, 'checked_rows', return_value=[{}]), patch.object(adapter, 'verify_code'):
            with patch.object(adapter, 'pipeline') as pipeline:
                with self.assertRaisesRegex(ValueError, message):
                    adapter.execute_admitted(contract, 'CPU_SYNTHETIC_NOT_A_REAL_CONTRACT', contextlib.nullcontext(resource))
                pipeline.assert_not_called()

    def resource(self):
        return {'parent_admitted': True, 'owner_lease_verified': True, 'gpu_index': 2,
                'gpu_uuid': 'GPU-CPU-SYNTHETIC', 'lease_expires_epoch': time.time() + 60}

    def test_missing_admission_rejected(self):
        self.reject(self.contract(), {}, 'Fresh parent/owner')

    def test_expired_lease_rejected(self):
        resource = self.resource()
        resource['lease_expires_epoch'] = time.time() - 1
        self.reject(self.contract(), resource, 'lease expired')

    def test_gpu0_rejected(self):
        resource = self.resource()
        resource['gpu_index'] = 0
        self.reject(self.contract(), resource, 'non-GPU0/7')

    def test_gpu7_rejected(self):
        resource = self.resource()
        resource['gpu_index'] = 7
        self.reject(self.contract(), resource, 'non-GPU0/7')

    def test_device_mismatch_rejected(self):
        resource = self.resource()
        resource['gpu_index'] = 4
        self.reject(self.contract(), resource, 'differs from owner')

    def test_missing_real_uid_rejected(self):
        resource = self.resource()
        resource['boot_id'] = 'CPU_SYNTHETIC_BOOT'
        self.reject(self.contract(), resource, 'UID/boot identity')

    def test_missing_real_boot_rejected(self):
        resource = self.resource()
        resource['uid'] = 999
        self.reject(self.contract(), resource, 'UID/boot identity')

    def test_standalone_contract_cannot_use_cluster_bridge(self):
        contract = self.contract()
        contract['execution_policy'] = 'standalone'
        self.reject(contract, self.resource(), 'Explicit cluster')


class LiveIdentityControls(unittest.TestCase):
    def setUp(self):
        self.resource = {'gpu_index': 2, 'gpu_uuid': 'GPU-CPU-SYNTHETIC',
                         'uid': 999, 'boot_id': 'CPU_SYNTHETIC_BOOT', 'lease_expires_epoch': 101}
        self.env = {'CUDA_VISIBLE_DEVICES': '2', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'R3_ADMITTED_GPU_UUID': 'GPU-CPU-SYNTHETIC'}

    def check(self, uid=999, boot='CPU_SYNTHETIC_BOOT', now=100, uuid='GPU-CPU-SYNTHETIC'):
        device.check_identity(self.resource, self.env, uid, boot, now, uuid)

    def test_synthetic_identity_positive(self):
        self.check()
        device.check_cuda_mapping(self.resource, 1, 'GPU-CPU-SYNTHETIC')

    def test_changed_uid_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BOOT_UID_CHANGED'):
            self.check(uid=888)

    def test_changed_boot_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BOOT_UID_CHANGED'):
            self.check(boot='DIFFERENT_SYNTHETIC_BOOT')

    def test_expired_actual_child_lease_rejected(self):
        with self.assertRaisesRegex(ValueError, 'OWNER_LEASE_EXPIRED'):
            self.check(now=101)

    def test_changed_physical_uuid_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PHYSICAL_INDEX_UUID_CHANGED'):
            self.check(uuid='GPU-DIFFERENT-SYNTHETIC')

    def test_numeric_mapping_rejected(self):
        self.env['CUDA_VISIBLE_DEVICES'] = '0'
        with self.assertRaisesRegex(ValueError, 'NUMERIC_VISIBILITY_MISMATCH'):
            self.check()

    def test_cuda_logical_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CUDA_ORDINAL_UUID_MISMATCH'):
            device.check_cuda_mapping(self.resource, 1, 'GPU-DIFFERENT-SYNTHETIC')


if __name__ == '__main__':
    unittest.main(verbosity=2)
