"""CPU-only fail-closed controls for the historical evidence-root API."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from assets import ROOT

spec=importlib.util.spec_from_file_location('historical_formal_gates',ROOT/'core/formal_gates.py')
gates=importlib.util.module_from_spec(spec);spec.loader.exec_module(gates)


class ExplicitFormalRoot(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'authorized';self.root.mkdir()
        self.evidence=self.root/'evidence.json';self.evidence.write_text(json.dumps({'synthetic_control':True}))
        self.ref={'path':str(self.evidence),'sha256':hashlib.sha256(self.evidence.read_bytes()).hexdigest()}

    def test_missing_caller_root_is_rejected(self):
        with self.assertRaises(TypeError):gates.acceptance({})

    def test_none_relative_or_file_root_fails_before_evidence_read(self):
        with patch.object(gates,'read_evidence',side_effect=AssertionError('Unexpected evidence read')):
            for root in (None,'relative',self.evidence,Path(self.root.anchor)):
                with self.subTest(root=root),self.assertRaises(ValueError):gates.acceptance({},root=root)

    def test_root_is_not_inferred_from_untrusted_contract(self):
        contract={'root':str(self.root),'bindings':{'evidence_root':str(self.root)}}
        with self.assertRaises(TypeError):gates.acceptance(contract)

    def test_authorized_root_retains_original_contract_gate(self):
        with self.assertRaisesRegex(ValueError,'FRESH_FORMAL_KIND_REQUIRED'):
            gates.acceptance({},root=self.root)

    def test_existing_boundary_and_sha_checks_remain(self):
        self.assertEqual(gates.read_evidence(self.ref,self.root),{'synthetic_control':True})
        outside=self.base/'outside.json';outside.write_bytes(self.evidence.read_bytes())
        with self.assertRaisesRegex(ValueError,'OUTSIDE_PROJECT_OR_LINK'):
            gates.read_evidence(dict(self.ref,path=str(outside)),self.root)
        self.evidence.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA_CHANGED'):gates.read_evidence(self.ref,self.root)


if __name__=='__main__':unittest.main()
