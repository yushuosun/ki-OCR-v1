"""Synthetic CPU trust-boundary controls; no model download or GPU work."""
import copy
import os
import hashlib
import json
import socket
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import assets
import offline_assets as offline
import offline_install
import install
from offline_import_guard import OfflineImportGuard

class OfflineAssets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(self.tmp.name)
        self.models = self.base/'public-models'
        self.manifest = {'schema':offline.SCHEMA, 'models':[]}
        self.pins = {}
        for kind,(repository,revision) in assets.MODELS.items():
            raw = ('SYNTHETIC_PUBLIC_FIXTURE_'+kind).encode()
            target = self.models/kind/'config.json'
            target.parent.mkdir(parents=True)
            target.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.pins[kind] = {'config.json':digest}
            self.manifest['models'].append({'kind':kind, 'repository':repository,
                'revision':revision, 'public_ungated':True, 'files':[{'file':'config.json',
                'bytes':len(raw), 'sha256':digest,
                'url':f'https://huggingface.co/{repository}/resolve/{revision}/config.json'}]})
        original_load = assets.load
        self.stack.enter_context(patch.object(offline, 'load', side_effect=lambda p:
            self.manifest if Path(p)==ROOT/'OFFLINE_MODEL_MANIFEST.json' else original_load(p)))
        self.stack.enter_context(patch.object(offline, 'model_pins', side_effect=lambda k:self.pins[k]))

    def supplied(self, value):
        target=self.base/'supplied.json'
        target.write_text(json.dumps(value),encoding='utf-8')
        return target

    def test_complete_preload_without_HF_import_network_or_token_lookup(self):
        import builtins
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('huggingface_hub','requests')):
                raise AssertionError('Offline path imported network/model SDK')
            return original_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=guarded), \
             patch.object(socket.socket,'connect',side_effect=AssertionError('Network touched')):
            result=offline.verify_preloaded_models(self.models)
        self.assertEqual(result['files_verified'],3)
        self.assertEqual(result['network_calls'],0)
        self.assertFalse(result['token_lookup'])

    def test_import_guard_blocks_and_records_even_caught_forbidden_calls(self):
        original_calls=[]
        def original(*args,**kwargs):original_calls.append(True)
        class API:
            model_info=original
        names=('huggingface_hub','huggingface_hub.hf_api','huggingface_hub.utils',
               'huggingface_hub.utils._auth','huggingface_hub.utils._headers','huggingface_hub.file_download')
        modules={n:types.ModuleType(n) for n in names}
        hf=modules['huggingface_hub'];hf.HfApi=API;hf.model_info=original
        hf.hf_hub_download=original;hf.snapshot_download=original
        for m in modules.values():m.get_token=original
        with patch.dict(sys.modules,modules):
            guard=OfflineImportGuard().install()
            try:
                for fn in (lambda:socket.create_connection(('example.invalid',80)),
                           lambda:hf.HfApi().model_info(),lambda:hf.get_token()):
                    with self.assertRaisesRegex(RuntimeError,'FORBIDDEN'):fn()
                with self.assertRaisesRegex(ValueError,'attempted forbidden'):guard.require_no_attempts()
                self.assertEqual(guard.attempts,{'network':1,'model_info_or_download':1,'token_lookup':1})
                self.assertEqual(original_calls,[])
            finally:guard.close()

    def test_wrong_size_fails_before_destination_creation(self):
        (self.models/'tele/config.json').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'size mismatch'):
            offline.import_preloaded_models(self.models,self.base/'install')
        self.assertFalse((self.base/'install').exists())

    def test_same_size_wrong_hash_rejected(self):
        target=self.models/'tele/config.json'
        target.write_bytes(b'x'*target.stat().st_size)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            offline.verify_preloaded_models(self.models)

    def test_missing_asset_rejected(self):
        (self.models/'paddle/config.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing/outside'):
            offline.verify_preloaded_models(self.models)

    def test_supplied_revision_cannot_override_fixed_model(self):
        value=copy.deepcopy(self.manifest);value['models'][0]['revision']='0'*40
        with self.assertRaisesRegex(ValueError,'revision mismatch'):
            offline.verify_preloaded_models(self.models,self.supplied(value))

    def test_supplied_size_cannot_override_frozen_inventory(self):
        value=copy.deepcopy(self.manifest);value['models'][0]['files'][0]['bytes']+=1
        with self.assertRaisesRegex(ValueError,'differs from frozen'):
            offline.verify_preloaded_models(self.models,self.supplied(value))

    def test_duplicate_manifest_file_rejected(self):
        value=copy.deepcopy(self.manifest);value['models'][0]['files']*=2
        with self.assertRaisesRegex(ValueError,'duplicate'):
            offline.verify_preloaded_models(self.models,self.supplied(value))

    def test_missing_manifest_model_rejected(self):
        value=copy.deepcopy(self.manifest);value['models'].pop()
        with self.assertRaisesRegex(ValueError,'All three'):
            offline.verify_preloaded_models(self.models,self.supplied(value))

    def test_path_traversal_manifest_rejected(self):
        value=copy.deepcopy(self.manifest);value['models'][0]['files'][0]['file']='../escape.json'
        with self.assertRaisesRegex(ValueError,'unsafe'):
            offline.verify_preloaded_models(self.models,self.supplied(value))

    def symlink(self,target,link,folder=False):
        if os.name=='nt':self.skipTest('Linux-target symlink fixture; no Windows privilege request')
        try:link.symlink_to(target,target_is_directory=folder)
        except (OSError,NotImplementedError):self.skipTest('Host cannot create a synthetic symlink')

    def test_outside_source_symlink_rejected(self):
        original=self.models/'tele/config.json'
        outside=self.base/'outside.json';outside.write_bytes(original.read_bytes());original.unlink()
        self.symlink(outside,original)
        with self.assertRaisesRegex(ValueError,'Missing/outside'):
            offline.verify_preloaded_models(self.models)

    def test_outside_destination_symlink_rejected(self):
        destination=self.base/'install';destination.mkdir()
        self.symlink(self.models,destination/'models',True)
        with self.assertRaisesRegex(ValueError,'symlink destination'):
            offline.import_preloaded_models(self.models,destination)

    def test_import_copies_only_allowlisted_files_and_preserves_bytes(self):
        cache=self.models/'.cache/token';cache.parent.mkdir();cache.write_bytes(b'SYNTHETIC_EXCLUDED_TOKEN')
        root=self.base/'install'
        result=offline.import_preloaded_models(self.models,root)
        self.assertEqual(result['copied_files'],3)
        self.assertFalse((root/'models/.cache').exists())
        for kind in assets.MODELS:
            self.assertEqual((root/'models'/kind/'config.json').read_bytes(),(self.models/kind/'config.json').read_bytes())

    def test_complete_destination_is_verified_and_reused(self):
        root=self.base/'install';offline.import_preloaded_models(self.models,root)
        result=offline.import_preloaded_models(self.models,root)
        self.assertEqual(result['copied_files'],0)
        self.assertEqual(result['status'],'OFFLINE_PUBLIC_MODELS_REUSED_VERIFIED')

    def test_incomplete_destination_is_not_merged_or_repaired(self):
        root=self.base/'install';(root/'models').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Missing/outside'):
            offline.import_preloaded_models(self.models,root)
        self.assertEqual(list((root/'models').iterdir()),[])

    def test_preloaded_installer_rejects_download_and_partial_components(self):
        with patch.object(install.sys,'platform','linux'), \
             patch.object(install.platform,'machine',return_value='x86_64'), \
             patch.object(install.sys,'version_info',(3,10,16)), \
             patch.object(install.venv,'EnvBuilder',side_effect=AssertionError('Reinstalled env')):
            with self.assertRaisesRegex(ValueError,'no download option'):
                install.install(self.base,['cpu','tele','paddle'],True,preloaded_models=self.models,reuse_environments=True)
            with self.assertRaisesRegex(ValueError,'all components'):
                install.install(self.base,['cpu'],False,preloaded_models=self.models)
            with self.assertRaisesRegex(ValueError,'no download/cache'):
                install.install(self.base,['cpu','tele','paddle'],False,pip_cache=self.base/'wheels',
                                preloaded_models=self.models,reuse_environments=True)

    def fresh_fixture(self, failed_command=None, bad_import=False):
        """Synthetic installer orchestration only; never an actual pip/import PASS."""
        calls=[]
        def native(root):
            (root/'native').mkdir()
        class Builder:
            def __init__(inner,**kwargs):
                self.assertFalse(kwargs['system_site_packages'])
                self.assertTrue(kwargs['with_pip'])
            def create(inner,target):
                executable=install.python_at(target)
                executable.parent.mkdir(parents=True)
                executable.touch()
                (target/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        def process(argv,**kwargs):
            calls.append((argv,kwargs['env']))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertNotIn('HF_TOKEN',kwargs['env'])
            if 'clone' in argv:
                Path(argv[-1]).mkdir()
            if '--output' in argv:
                target=Path(argv[argv.index('--output')+1])
                payload={'status':'CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE',
                    'strict_offline':True,'offline_guard':{'attempts':
                    {'network':int(bad_import),'model_info_or_download':0,'token_lookup':0}}}
                target.write_text(json.dumps(payload))
            code=7 if failed_command and failed_command in argv else 0
            return types.SimpleNamespace(pid=123456,returncode=code,wait=lambda:code)
        self.stack.enter_context(patch.object(install.sys,'platform','linux'))
        self.stack.enter_context(patch.object(install.platform,'machine',return_value='x86_64'))
        self.stack.enter_context(patch.object(install.sys,'version_info',(3,10,16)))
        self.stack.enter_context(patch('native_setup.setup',side_effect=native))
        self.stack.enter_context(patch.object(install,'verify_pins'))
        self.stack.enter_context(patch.object(install.venv,'EnvBuilder',Builder))
        self.stack.enter_context(patch.object(install.subprocess,'Popen',side_effect=process))
        return calls

    def test_fresh_preloaded_creates_three_envs_copies_models_and_new_runtime(self):
        calls=self.fresh_fixture()
        root=self.base/'fresh'
        old=self.base/'old-install';old.mkdir()
        sentinel=old/'runtime.json';sentinel.write_bytes(b'SYNTHETIC_OLD_RUNTIME')
        wheels=self.base/'public-wheel-cache';wheels.mkdir()
        with patch.dict(os.environ,{'HF_TOKEN':'SYNTHETIC_MUST_NOT_PROPAGATE',
                                    'VIRTUAL_ENV':str(old)}):
            result=install.install(root,['cpu','tele','paddle'],False,pip_cache=wheels,
                                   preloaded_models=self.models)
        self.assertFalse(result['environment_reused'])
        self.assertEqual(result['model_result']['copied_files'],3)
        self.assertTrue(result['dependency_downloads_online'])
        self.assertFalse(result['model_HF_requests'])
        self.assertFalse(result['GPU_started'])
        self.assertEqual(sentinel.read_bytes(),b'SYNTHETIC_OLD_RUNTIME')
        for kind in ('cpu','tele','paddle'):
            self.assertIn('false',(root/(kind+'_env')/'pyvenv.cfg').read_text())
        runtime=assets.load(root/'runtime.json')
        self.assertEqual(runtime['execution_policy'],'standalone')
        self.assertEqual(runtime['native_prefix'],str(root/'native'))
        self.assertTrue(all(str(old) not in str(v) for v in runtime.values()))
        imports=[(a,e) for a,e in calls if '--output' in a]
        self.assertEqual(len(imports),2)
        for argv,env in imports:
            self.assertIn('--strict-offline',argv)
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'],'True')
        self.assertFalse(any('--download' in a for a,e in calls))
        self.assertTrue(any('git'==a[0] and 'clone' in a for a,e in calls))
        self.assertTrue(any(str(wheels) in a for a,e in calls))

    def test_fresh_preloaded_accepts_existing_empty_root(self):
        self.fresh_fixture();root=self.base/'empty';root.mkdir()
        self.assertFalse(install.install(root,['cpu','tele','paddle'],False,
                         preloaded_models=self.models)['environment_reused'])

    def test_fresh_preloaded_rejects_nonempty_root_before_installs(self):
        calls=self.fresh_fixture();root=self.base/'old-root';root.mkdir()
        marker=root/'runtime.json';marker.write_bytes(b'SYNTHETIC_EXISTING_DESCRIPTOR')
        with self.assertRaisesRegex(ValueError,'completely empty root'):
            install.install(root,['cpu','tele','paddle'],False,preloaded_models=self.models)
        self.assertEqual(calls,[])
        self.assertEqual(marker.read_bytes(),b'SYNTHETIC_EXISTING_DESCRIPTOR')

    def test_fresh_preloaded_rejects_external_native_prefix(self):
        calls=self.fresh_fixture()
        with self.assertRaisesRegex(ValueError,'own native runtime'):
            install.install(self.base/'new',['cpu','tele','paddle'],False,
                            preloaded_models=self.models,native_prefix=self.base/'old-native')
        self.assertEqual(calls,[])

    def test_fresh_preloaded_checks_models_before_install_side_effects(self):
        calls=self.fresh_fixture();root=self.base/'new'
        (self.models/'tele/config.json').write_bytes(b'BAD')
        with self.assertRaisesRegex(ValueError,'size mismatch'):
            install.install(root,['cpu','tele','paddle'],False,preloaded_models=self.models)
        self.assertEqual(calls,[])
        self.assertFalse(root.exists())

    def test_fresh_dependency_failure_stops_without_runtime_or_model_copy(self):
        calls=self.fresh_fixture(failed_command='clone');root=self.base/'failed'
        with self.assertRaisesRegex(RuntimeError,'tele_source_clone'):
            install.install(root,['cpu','tele','paddle'],False,preloaded_models=self.models)
        self.assertEqual(assets.load(root/'INSTALL_RESULT.json')['commands'][-1]['exit'],7)
        self.assertFalse((root/'runtime.json').exists())
        self.assertFalse((root/'models').exists())
        self.assertFalse((root/'paddle_env').exists())

    def test_fresh_forbidden_CPU_import_attempt_stops_model_copy(self):
        self.fresh_fixture(bad_import=True);root=self.base/'failed'
        with self.assertRaisesRegex(ValueError,'offline CPU import did not pass'):
            install.install(root,['cpu','tele','paddle'],False,preloaded_models=self.models)
        self.assertFalse((root/'runtime.json').exists())
        self.assertFalse((root/'models').exists())

    def test_default_online_route_keeps_HF_download_and_default_imports(self):
        calls=self.fresh_fixture();root=self.base/'online'
        result=install.install(root,['cpu','tele','paddle'],True)
        self.assertTrue(result['model_download_requested'])
        self.assertTrue(any('--download' in a for a,e in calls))
        self.assertFalse(any('--strict-offline' in a for a,e in calls))
        self.assertNotIn('model_provisioning',result)

    def fake_install(self):
        root=self.base/'installed';root.mkdir()
        (root/'TeleOCR').mkdir();(root/'native').mkdir()
        for kind in ('cpu','tele','paddle'):
            executable=root/(kind+'_env')/'bin/python'
            executable.parent.mkdir(parents=True);executable.touch()
        return root

    def test_missing_existing_environment_stops_without_network_repair(self):
        root=self.fake_install();(root/'cpu_env/bin/python').unlink()
        with patch.object(offline_install,'verify_pins'), \
             patch('inference.verify_environment') as metadata, \
             patch.object(offline_install.subprocess,'run',side_effect=AssertionError('Repair attempted')):
            with self.assertRaisesRegex(ValueError,'Complete existing isolated'):
                offline_install.reuse_with_preloaded(root,self.models)
            metadata.assert_not_called()

    def test_reuse_checks_all_envs_and_actual_imports_without_pip_or_Git(self):
        root=self.fake_install();calls=[]
        def imported(argv,**kwargs):
            self.assertIn('import_preflight.py',str(argv[2]))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
            self.assertNotIn('pip',argv);self.assertNotIn('git',argv)
            calls.append(argv)
            self.assertIn('--strict-offline',argv)
            Path(argv[argv.index('--output')+1]).write_text(json.dumps({'status':'CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE',
                'strict_offline':True,'offline_guard':{'attempts':{'network':0,'model_info_or_download':0,'token_lookup':0}}}))
            return types.SimpleNamespace(returncode=0)
        with patch.object(offline_install,'verify_pins'), \
             patch('inference.verify_environment',return_value={'synthetic':True}) as metadata, \
             patch.object(offline_install.subprocess,'run',side_effect=imported):
            result=offline_install.reuse_with_preloaded(root,self.models)
        self.assertEqual([c.args[1] for c in metadata.call_args_list],['cpu','tele','paddle'])
        self.assertEqual(len(calls),2)
        self.assertFalse(result['fresh_uninterrupted_install'])
        self.assertFalse(result['GPU_started'])
        self.assertEqual(result['network_calls'],0)

if __name__=='__main__':unittest.main()
