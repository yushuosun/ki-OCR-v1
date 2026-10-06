"""One explicit continuation of an R5 Tele clone failure; no GPU admission.

Other failure stages stop rather than guessing how to repair partial installs.
Original receipts, logs, environments and failed clone directories stay intact.
"""
import json
import subprocess
import time
import uuid
from pathlib import Path
from assets import ROOT, clean_env, load, sha, verify_pins, write

R5_FILES_SHA = 'fe2b92993a8e4db19f6662d8a92ee2e9d868726099c49203e80e8755ce7563bf'
SUCCESS_STAGES = ('cpu_install', 'cpu_pip_check', 'cpu_environment', 'tele_install')
CURRENT_RESUME = None


def checked_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Receipt path must remain inside installation root')
    return path


def verify_dependency_env(root, kind):
    """Check an installed dependency lock before Tele's plugin exists."""
    target = root / (kind + '_env')
    config = target / 'pyvenv.cfg'
    if target.is_symlink() or config.is_symlink() or not config.is_file():
        raise ValueError('Actual new-root venv config required: ' + kind)
    if 'include-system-site-packages = false' not in config.read_text().lower():
        raise ValueError('Isolated environment required: ' + kind)
    python = target / 'bin/python'
    script = ("import sys,json,importlib.metadata as m;print(json.dumps({'prefix':sys.prefix,"
              "'base_prefix':sys.base_prefix,'python':list(sys.version_info[:3]),"
              "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))")
    env = clean_env(root, root / 'native/lib');env['CUDA_VISIBLE_DEVICES'] = ''
    identity = json.loads(subprocess.check_output([str(python), '-I', '-c', script],
                          env=env, text=True, timeout=60))
    if (Path(identity['prefix']).resolve() != target.resolve()
            or identity['prefix'] == identity['base_prefix'] or identity['python'][:2] != [3, 10]):
        raise ValueError('Actual interpreter must belong to this new-root venv: ' + kind)
    expected = dict(line.strip().split('==') for line in
                    (ROOT / ('requirements-' + kind + '.lock')).read_text().splitlines()
                    if line.strip() and not line.startswith('#'))
    from inference import check_locked_versions
    count = check_locked_versions(identity['packages'], expected, kind)
    return {'prefix':identity['prefix'], 'base_prefix':identity['base_prefix'],
            'locked_versions_verified':count, 'plugin_not_yet_required':kind == 'tele',
            'GPU_started':False}


class CloneResume:
    def __init__(self, root, receipt_sha, pip_cache, timeout_seconds=10800):
        self.started = time.monotonic()
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 10800:
            raise ValueError('Explicit resume wall bound must be 1..10800 seconds')
        self.deadline = self.started+timeout_seconds
        self.root = Path(root).absolute()
        if (self.root.is_symlink() or not self.root.is_dir() or (self.root/'runtime.json').exists()
                or (self.root/'paddle_env').exists() or (self.root/'models').exists()):
            raise ValueError('Only incomplete R5 Tele-clone checkpoint can be resumed')
        self.initial = self.root/'INSTALL_RESULT.json'
        if self.initial.is_symlink() or sha(self.initial) != receipt_sha:
            raise ValueError('Original installation receipt SHA mismatch')
        prior = load(self.initial)
        rows = prior.get('commands', [])
        if (prior.get('GPU_started') is not False or prior.get('active_pid')
                or len(rows) != 5 or [r.get('log') for r in rows] !=
                [name+'.log' for name in (*SUCCESS_STAGES, 'tele_source_clone')]
                or any(type(r.get('exit')) is not int or r['exit'] != 0 for r in rows[:4])
                or type(rows[4].get('exit')) is not int or rows[4]['exit'] == 0):
            raise ValueError('Require four successful R5 steps and a terminal failed Tele clone')
        for row in rows:
            log = checked_path(self.root, row['log'])
            if not log.is_file() or sha(log) != row.get('log_sha256'):
                raise ValueError('Original command log SHA mismatch: ' + row['log'])
        first = rows[0]['argv']
        if len(first) < 2 or first[-2] != '-r':
            raise ValueError('Original frozen requirements argv required')
        old_source = Path(first[-1]).parent
        if sha(old_source/'FILES.json') != R5_FILES_SHA:
            raise ValueError('Original source must match frozen R5 manifest')
        verify_pins(old_source, load(old_source/'FILES.json'))
        # The resume candidate may change installer/docs/tests only, never assets.
        for name in ('requirements-cpu.lock','requirements-tele.lock','requirements-paddle.lock',
                     'DEPENDENCIES.json','UPSTREAM_PLUGIN_PINS.json','NATIVE_RUNTIME_LOCK.json',
                     'PADDLE_WHEEL_PIN.json','OFFLINE_MODEL_MANIFEST.json','MODEL_PINS.json',
                     'PADDLE_MODEL_PINS.json'):
            if sha(ROOT/name) != sha(old_source/name):
                raise ValueError('Frozen installation contract changed: ' + name)
        common = ['-m','pip','--isolated','--timeout','300','--retries','0']
        expected = []
        for kind in ('cpu','tele'):
            py = str(self.root/(kind+'_env')/'bin/python')
            expected.append([py,*common,'install','--index-url','https://pypi.org/simple',
                '--cache-dir',str(pip_cache),'-r',str(old_source/('requirements-'+kind+'.lock'))])
            if kind == 'cpu':
                expected.append([py,*common,'check'])
                expected.append([py,'-I','-c',
                    "import json,sys,importlib.metadata as m; print(json.dumps({'prefix':sys.prefix,"
                    "'base_prefix':sys.base_prefix,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"])
        deps = load(ROOT/'DEPENDENCIES.json')
        expected.append(['git','clone','--no-checkout',deps['tele_source_url'],str(self.root/'TeleOCR')])
        if [r.get('argv') for r in rows] != expected:
            raise ValueError('Original command argv does not match frozen R5 checkpoint')
        native_receipt = self.root/'NATIVE_INSTALL_RESULT.json'
        if native_receipt.is_symlink() or (self.root/'native').is_symlink():
            raise ValueError('Native runtime and receipt must stay inside original root')
        native = load(native_receipt)
        lock = load(ROOT/'NATIVE_RUNTIME_LOCK.json')
        if (native.get('status') != 'PUBLIC_NATIVE_RUNTIME_INSTALLED'
                or native.get('GPU_started') is not False or native.get('system_changed') is not False
                or native.get('archives') != [{'sha256':p['sha256'],'package':p['name'],'version':p['version']}
                                             for p in lock['packages']]):
            raise ValueError('Successful frozen native receipt required')
        verify_pins(self.root/'native', lock['library_pins'])
        self.environments = {k:verify_dependency_env(self.root,k) for k in ('cpu','tele')}
        self.require_time()
        parent = self.root/'installer-runs'
        if parent.is_symlink() or (parent.exists() and any(parent.glob('resume-*'))):
            raise ValueError('Only one explicit resume attempt per initial failed root')
        parent.mkdir(exist_ok=True)
        # Exclusive claim also rejects concurrent calls before either clone starts.
        with (parent/'RESUME_ONCE.json').open('x', encoding='utf-8') as stream:
            json.dump({'original_receipt_sha256':receipt_sha,'maximum_attempts':1},stream)
        self.run = parent/('resume-'+uuid.uuid4().hex)
        self.run.mkdir(parents=True, exist_ok=False)
        self.source = self.run/'TeleOCR'
        self.result_path = self.run/'INSTALL_RESULT.json'
        self.rows = {stage:row for stage,row in zip(SUCCESS_STAGES,rows[:4])}
        self.old_source = old_source
        self.receipt_sha = receipt_sha
        self.original_rows = rows
        global CURRENT_RESUME
        CURRENT_RESUME = self

    def require_time(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Explicit resume wall bound reached; no automatic retry')

    def skipped(self, name, argv):
        if name not in self.rows:
            return None
        old = self.rows[name]
        mapped = [str(self.old_source/Path(x).relative_to(ROOT))
                  if Path(x).is_absolute() and Path(x).is_relative_to(ROOT) else x for x in argv]
        if mapped != old['argv']:
            raise ValueError('Cannot skip changed command: '+name)
        return dict(old, stage=name, resumed_skip=True, original_receipt_sha256=self.receipt_sha)

    def facts(self):
        return {'resumed':True,'uninterrupted':False,'fresh_uninterrupted_install':False,
                'original_install_result':str(self.initial),'original_install_result_sha256':self.receipt_sha,
                'initial_failed_commands':self.original_rows,'verification_run':str(self.run),
                'resume_result':str(self.result_path),'tele_source':str(self.source),
                'verified_existing_dependency_environments':self.environments,
                'initial_empty_root_evidence':'Separate original owner evidence required; not inferred',
                'automatic_retry':False,'maximum_resume_attempts':1,'git_http_version':'HTTP/1.1',
                'HTTP1_clone_success':None,
                'GPU_started':False,'admission_or_registration':False}

    def cost(self, commands):
        write(self.run/'COST.json',{'resume_wall_seconds':time.monotonic()-self.started,
            'GPU_card_seconds':0,'initial_failed_commands':1,'resume_attempts':1,
            'resume_failed_commands':sum(r.get('exit',0)!=0 for r in commands),
            'automatic_retry':False,'full_model_GPU_inference':'NOT_RUN'})

    def failed(self, error):
        value = load(self.result_path) if self.result_path.exists() else {}
        for key in ('active_pid','active_argv','active_log'):
            value.pop(key,None)
        value.update(self.facts(),status='RESUME_FAILED',error_type=type(error).__name__,error=str(error))
        clone = next((r for r in value.get('commands',[]) if r.get('stage')=='tele_source_clone'),None)
        if clone is not None:
            value['HTTP1_clone_success'] = clone['exit']==0 and not clone.get('timeout_reached')
        write(self.result_path,value)
        self.cost(value.get('commands',[]))
