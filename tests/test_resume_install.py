"""CPU synthetic checkpoint controls, not a real network/install acceptance."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import assets
import install
import resume_install as resume


class ResumeControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name);self.root = self.base/'new-install';self.root.mkdir()
        self.contract = self.base/'candidate';self.contract.mkdir()
        self.old = self.base/'frozen-r5';self.old.mkdir()
        names = ('requirements-cpu.lock','requirements-tele.lock','requirements-paddle.lock',
                 'DEPENDENCIES.json','UPSTREAM_PLUGIN_PINS.json','NATIVE_RUNTIME_LOCK.json',
                 'PADDLE_WHEEL_PIN.json','OFFLINE_MODEL_MANIFEST.json','MODEL_PINS.json','PADDLE_MODEL_PINS.json')
        for name in names:
            (self.contract/name).write_bytes((assets.ROOT/name).read_bytes())
        lock = assets.load(self.contract/'NATIVE_RUNTIME_LOCK.json')
        lock['library_pins'] = {'lib/libgomp.so.1':assets.sha(self.contract/'DEPENDENCIES.json')}
        assets.write(self.contract/'NATIVE_RUNTIME_LOCK.json',lock)
        for name in names:(self.old/name).write_bytes((self.contract/name).read_bytes())
        assets.write(self.old/'FILES.json',{n:assets.sha(self.old/n) for n in names})
        (self.root/'native/lib').mkdir(parents=True)
        (self.root/'native/lib/libgomp.so.1').write_bytes((self.contract/'DEPENDENCIES.json').read_bytes())
        assets.write(self.root/'NATIVE_INSTALL_RESULT.json',{'status':'PUBLIC_NATIVE_RUNTIME_INSTALLED',
            'GPU_started':False,'system_changed':False,
            'archives':[{'sha256':p['sha256'],'package':p['name'],'version':p['version']} for p in lock['packages']]})
        self.identities = {}
        for kind in ('cpu','tele'):
            env=self.root/(kind+'_env');(env/'bin').mkdir(parents=True);(env/'bin/python').touch()
            (env/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            packages=dict(line.split('==') for line in (self.contract/('requirements-'+kind+'.lock')).read_text().splitlines()
                          if line and not line.startswith('#'))
            self.identities[kind]={'prefix':str(env),'base_prefix':str(self.base/'base'),'python':[3,10,16],'packages':packages}
        common=['-m','pip','--isolated','--timeout','300','--retries','0']
        rows=[]
        for kind in ('cpu','tele'):
            py=str(self.root/(kind+'_env')/'bin/python')
            rows.append((kind+'_install',[py,*common,'install','--index-url','https://pypi.org/simple','--cache-dir',
                str(self.root/'pip-cache'),'-r',str(self.old/('requirements-'+kind+'.lock'))],0))
            if kind=='cpu':
                rows.append(('cpu_pip_check',[py,*common,'check'],0))
                rows.append(('cpu_environment',[py,'-I','-c',
                    "import json,sys,importlib.metadata as m; print(json.dumps({'prefix':sys.prefix,"
                    "'base_prefix':sys.base_prefix,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"],0))
        rows.append(('tele_source_clone',['git','clone','--no-checkout',
            assets.load(self.contract/'DEPENDENCIES.json')['tele_source_url'],str(self.root/'TeleOCR')],128))
        self.rows=[]
        for name,argv,code in rows:
            log=self.root/(name+'.log');log.write_text('synthetic '+name+' '+str(code)+'\n')
            self.rows.append({'argv':argv,'exit':code,'log':log.name,'log_sha256':assets.sha(log),'seconds':0.01})
        assets.write(self.root/'INSTALL_RESULT.json',{'commands':self.rows,'GPU_started':False})
        (self.root/'TeleOCR').mkdir();(self.root/'TeleOCR/partial').write_bytes(b'FAILED_CLONE_PRESERVE')
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(resume,'ROOT',self.contract))
        self.stack.enter_context(patch.object(install,'ROOT',self.contract))
        self.stack.enter_context(patch.object(resume,'R5_FILES_SHA',assets.sha(self.old/'FILES.json')))
        def metadata(argv,**kwargs):
            kind='tele' if 'tele_env' in str(argv[0]) else 'cpu'
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            return json.dumps(self.identities[kind])
        self.stack.enter_context(patch.object(resume.subprocess,'check_output',side_effect=metadata))
        resume.CURRENT_RESUME=None

    def state(self, digest=None):
        return resume.CloneResume(self.root,digest or assets.sha(self.root/'INSTALL_RESULT.json'),self.root/'pip-cache')

    def test_real_hashes_and_locks_allow_only_four_skips_without_plugin(self):
        state=self.state()
        self.assertEqual(state.environments['cpu']['locked_versions_verified'],15)
        self.assertEqual(state.environments['tele']['locked_versions_verified'],179)
        self.assertTrue(state.environments['tele']['plugin_not_yet_required'])
        self.assertEqual(tuple(state.rows),resume.SUCCESS_STAGES)
        self.assertEqual((self.root/'TeleOCR/partial').read_bytes(),b'FAILED_CLONE_PRESERVE')
        self.assertTrue(state.source.is_relative_to(state.run))
        self.assertTrue(state.facts()['resumed']);self.assertFalse(state.facts()['uninterrupted'])

    def test_wrong_receipt_sha_rejected(self):
        with self.assertRaisesRegex(ValueError,'receipt SHA'):self.state('0'*64)
        self.assertFalse((self.root/'installer-runs').exists())

    def test_modified_success_log_rejected(self):
        (self.root/'tele_install.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'log SHA'):self.state()

    def test_wrong_actual_prefix_rejected(self):
        self.identities['tele']['prefix']=str(self.base/'old-env')
        with self.assertRaisesRegex(ValueError,'this new-root'):self.state()

    def test_missing_dependency_rejected_without_directory_based_skip(self):
        self.identities['tele']['packages'].pop('vllm')
        with self.assertRaisesRegex(ValueError,'dependency missing/changed'):self.state()

    def test_changed_native_pin_rejected(self):
        (self.root/'native/lib/libgomp.so.1').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Pinned asset SHA'):self.state()

    def test_forged_success_checkpoint_rejected(self):
        data=assets.load(self.root/'INSTALL_RESULT.json');data['commands'][-1]['exit']=0
        assets.write(self.root/'INSTALL_RESULT.json',data)
        with self.assertRaisesRegex(ValueError,'terminal failed Tele clone'):self.state()

    def test_other_command_argv_rejected(self):
        data=assets.load(self.root/'INSTALL_RESULT.json');data['commands'][3]['argv'][-1]='/wrong/requirements-tele.lock'
        assets.write(self.root/'INSTALL_RESULT.json',data)
        with self.assertRaisesRegex(ValueError,'argv'):self.state()

    def test_single_resume_and_changed_skip_command_rejected(self):
        state=self.state()
        with self.assertRaisesRegex(ValueError,'changed command'):state.skipped('tele_install',['changed'])
        with self.assertRaisesRegex(ValueError,'Only one'):self.state()

    def test_new_attempt_nonzero_failure_preserves_original(self):
        original=(self.root/'INSTALL_RESULT.json').read_bytes();state=self.state()
        state.failed(RuntimeError('HTTP1.1 clone failed: synthetic'))
        self.assertEqual((self.root/'INSTALL_RESULT.json').read_bytes(),original)
        self.assertEqual(assets.load(state.result_path)['status'],'RESUME_FAILED')
        self.assertFalse(assets.load(state.result_path)['uninterrupted'])
        self.assertFalse((self.root/'runtime.json').exists())

    def test_expired_checkpoint_bound_stops_without_new_actions(self):
        state=self.state();state.deadline=0
        with self.assertRaisesRegex(TimeoutError,'wall bound'):state.require_time()
        self.assertFalse((self.root/'runtime.json').exists())
        self.assertFalse((self.root/'paddle_env').exists())

    def test_actual_cli_control_records_failed_clone_without_retry_or_runtime(self):
        original=(self.root/'INSTALL_RESULT.json').read_bytes();calls=[]
        def child(argv,**kwargs):
            calls.append(argv)
            self.assertIn('clone',argv)
            return types.SimpleNamespace(pid=123456,returncode=128,wait=lambda **kw:128)
        argv=['install.py','--root',str(self.root),'--preloaded-models',str(self.base/'models'),
              '--resume-install','--resume-receipt-sha256',assets.sha(self.root/'INSTALL_RESULT.json')]
        with contextlib.redirect_stdout(io.StringIO()),patch.object(install.sys,'argv',argv), \
             patch.object(install.sys,'platform','linux'),patch.object(install.sys,'version_info',(3,10,16)), \
             patch.object(install.platform,'machine',return_value='x86_64'), \
             patch.object(install.subprocess,'Popen',side_effect=child), \
             patch('offline_assets.verify_preloaded_models'):
            with self.assertRaises(SystemExit) as caught:install.main()
        self.assertEqual(caught.exception.code,2);self.assertEqual(len(calls),1)
        state=resume.CURRENT_RESUME;result=assets.load(state.result_path)
        self.assertEqual(result['status'],'RESUME_FAILED');self.assertFalse(result['HTTP1_clone_success'])
        self.assertEqual((self.root/'INSTALL_RESULT.json').read_bytes(),original)
        self.assertEqual((self.root/'TeleOCR/partial').read_bytes(),b'FAILED_CLONE_PRESERVE')
        self.assertFalse((self.root/'runtime.json').exists())

    def test_installer_continuation_skips_locks_clones_once_and_binds_new_source(self):
        original=(self.root/'INSTALL_RESULT.json').read_bytes();calls=[]
        class Builder:
            def __init__(inner,**kw):pass
            def create(inner,target):
                self.assertEqual(target.name,'paddle_env')
                (target/'bin').mkdir(parents=True);(target/'bin/python').touch()
                (target/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        def child(argv,**kwargs):
            calls.append(argv);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertTrue(kwargs.get('start_new_session'))
            if 'clone' in argv:Path(argv[-1]).mkdir()
            if '--output' in argv:
                assets.write(Path(argv[argv.index('--output')+1]),{'status':'CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE',
                    'strict_offline':True,'offline_guard':{'attempts':{'network':0,'model_info_or_download':0,'token_lookup':0}}})
            return types.SimpleNamespace(pid=123456,returncode=0,wait=lambda **kw:0)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(install.sys,'platform','linux'),patch.object(install.sys,'version_info',(3,10,16)), \
             patch.object(install.platform,'machine',return_value='x86_64'), \
             patch.object(install.venv,'EnvBuilder',Builder),patch.object(install.subprocess,'Popen',side_effect=child), \
             patch.object(install,'verify_pins'),patch('offline_assets.verify_preloaded_models'), \
             patch('offline_assets.import_preloaded_models',return_value={'copied_files':32,'files_verified':32}):
            result=install.install(self.root,['cpu','tele','paddle'],False,preloaded_models=self.base/'public-models',
                resume_install=True,resume_receipt_sha256=assets.sha(self.root/'INSTALL_RESULT.json'))
        clones=[a for a in calls if 'clone' in a]
        self.assertEqual(len(clones),1);self.assertEqual(clones[0][:4],['git','-c','http.version=HTTP/1.1','clone'])
        self.assertEqual(len(calls),11)
        self.assertEqual((self.root/'INSTALL_RESULT.json').read_bytes(),original)
        self.assertEqual((self.root/'TeleOCR/partial').read_bytes(),b'FAILED_CLONE_PRESERVE')
        runtime=assets.load(self.root/'runtime.json')
        self.assertNotEqual(runtime['tele_source'],str(self.root/'TeleOCR'))
        self.assertTrue(result['resumed']);self.assertFalse(result['uninterrupted'])
        self.assertTrue(result['HTTP1_clone_success']);self.assertTrue(result['environment_reused'])
        self.assertFalse(result['external_environment_reused'])
        self.assertEqual(sum(r.get('resumed_skip',False) for r in result['commands']),4)


if __name__=='__main__':unittest.main()
