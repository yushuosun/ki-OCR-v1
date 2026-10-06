"""Negative controls for the actual installed plugin and complete archive."""
import json,sys,tempfile,unittest,zipfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from assets import load
from plugin_audit import check_binding
from archive_audit import audit

class ReleaseGuards(unittest.TestCase):
    def binding(self):
        return {'top_level_version':'0.1.0','independent_plugin_installed':False,
                'entrypoints':['TeleOCR_vllm:register'],
                'files':{n.removeprefix('TeleOCR-vllm/'):v for n,v in load(ROOT/'UPSTREAM_PLUGIN_PINS.json').items()}}
    def test_independent_plugin_mixed_install_is_rejected(self):
        value=self.binding();value['independent_plugin_installed']=True
        with self.assertRaisesRegex(ValueError,'independent plugin'):check_binding(value)
    def test_changed_actual_plugin_file_is_rejected(self):
        value=self.binding();value['files']['TeleOCR_vllm/qwen2_5_vl.py']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA differs'):check_binding(value)
    def test_conflicting_source_metadata_is_rejected(self):
        value=self.binding();value['metadata_conflict']=True
        with self.assertRaisesRegex(ValueError,'Conflicting Tele'):check_binding(value)
    def test_approved_top_level_plugin_binding(self):
        self.assertEqual(check_binding(self.binding())['plugin_files_SHA_verified'],2)
    def test_extra_model_body_in_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'synthetic.zip';raw=b'{}'
            with zipfile.ZipFile(p,'w') as stream:
                stream.writestr('FILES.json',raw);stream.writestr('models/private.bin',b'SYNTHETIC_PROHIBITED_BODY')
            with self.assertRaisesRegex(ValueError,'Unlisted ZIP file'):
                audit(p,{},hashlib.sha256(raw).hexdigest())
    def test_duplicate_archive_member_is_rejected(self):
        import warnings
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'synthetic.zip';raw=b'{}'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(p,'w') as stream:
                    stream.writestr('FILES.json',raw);stream.writestr('FILES.json',raw)
            with self.assertRaisesRegex(ValueError,'Duplicate/unsafe'):
                audit(p,{},hashlib.sha256(raw).hexdigest())

if __name__=='__main__':unittest.main()
