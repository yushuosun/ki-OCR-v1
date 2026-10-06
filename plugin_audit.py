"""Bind the actually installed and imported public Tele vLLM model plugin."""
import importlib,importlib.metadata as metadata,importlib.util,importlib.machinery,sys,sysconfig
from pathlib import Path
from assets import ROOT,load,sha

def check_binding(report):
    if report.get('metadata_conflict'):
        raise ValueError('Conflicting Tele source/distribution registration metadata.')
    if report['top_level_version']!='0.1.0' or report['independent_plugin_installed']:
        raise ValueError('Use only pinned top-level TeleOCR0.1.0 installation, not independent plugin.')
    if report['entrypoints']!=['TeleOCR_vllm:register']:
        raise ValueError('Actual Tele vLLM registration is missing/ambiguous.')
    expected={name.removeprefix('TeleOCR-vllm/'):pin for name,pin in load(ROOT/'UPSTREAM_PLUGIN_PINS.json').items()}
    if report['files']!=expected:
        raise ValueError('Actual installed Tele vLLM plugin SHA differs from public pins.')
    return {'installation_route':'top-level TeleOCR0.1.0 --no-deps',
            'plugin_files_SHA_verified':len(expected),'independent_plugin_used':False}

def inspect_installed():
    normalize=lambda s:s.lower().replace('_','-')
    site=Path(sysconfig.get_path('purelib')).resolve()
    distributions=list(metadata.distributions(path=[str(site)]))
    tops=[d for d in distributions if normalize(d.metadata['Name'])=='teleocr']
    if len(tops)!=1:raise ValueError('Actual installed top-level Tele distribution missing/ambiguous.')
    top=tops[0]
    # A public source clone can also contain its build-generated egg-info.
    # Use the same deduplicating entry-point API as vLLM, while auditing the
    # actual installed distribution separately; do not count a clone twice.
    entries=[entry.value for entry in metadata.entry_points(group='vllm.general_plugins')
             if entry.name=='TeleOCR_vllm' or entry.value.startswith('TeleOCR_vllm:')]
    spec=importlib.util.find_spec('TeleOCR_vllm')
    if spec is None or spec.origin is None or Path(spec.origin).resolve()!=site/'TeleOCR_vllm/__init__.py':
        raise ValueError('Actual Tele plugin resolves outside this isolated venv.')
    expected=load(ROOT/'UPSTREAM_PLUGIN_PINS.json');files={}
    for name in expected:
        relative=name.removeprefix('TeleOCR-vllm/');path=site/relative
        if not path.is_file() or not path.resolve().is_relative_to(site):
            raise ValueError('Missing/outside actual installed Tele plugin: '+relative)
        files[relative]=sha(path)
    report={'top_level_version':top.version,'entrypoints':sorted(entries),'files':files,
            'independent_plugin_installed':any(normalize(d.metadata['Name'])=='teleocr-vllm' for d in distributions),
            'top_level_transformers_requirement':[s for s in (top.requires or []) if s.lower().startswith('transformers')],
            'installed_root':str(site/'TeleOCR_vllm')}
    installed_entry_file=Path(top._path)/'entry_points.txt'
    duplicates=[]
    for dist in metadata.distributions():
        if normalize(dist.metadata['Name'])!='teleocr':continue
        entry_file=Path(dist._path)/'entry_points.txt'
        duplicates.append({'path':str(dist._path),'version':dist.version,'entrypoints_sha256':sha(entry_file)})
        if dist.version!=top.version or sha(entry_file)!=sha(installed_entry_file):
            report['metadata_conflict']=True
    report['same_source_metadata']=duplicates
    report.update(check_binding(report))
    return report

def audit_loaded():
    report=inspect_installed();root=Path(report['installed_root']);records=[]
    for name,filename in [('TeleOCR_vllm','__init__.py'),('TeleOCR_vllm.qwen2_5_vl','qwen2_5_vl.py')]:
        module=sys.modules.get(name);expected=root/filename
        if module is None:
            raise ValueError('Required actual Tele plugin module not loaded: '+name)
        spec=getattr(module,'__spec__',None);loader=getattr(module,'__loader__',None)
        if (spec is None or spec.name!=name or spec.loader is not loader
                or type(loader) is not importlib.machinery.SourceFileLoader
                or Path(module.__file__).resolve()!=expected
                or Path(spec.origin).resolve()!=expected):
            raise ValueError('Actual loaded Tele plugin module source/loader mismatch: '+name)
        records.append({'module':name,'path':str(expected),'sha256':sha(expected)})
    extras=[name for name in sys.modules if name.startswith('TeleOCR_vllm.') and name!='TeleOCR_vllm.qwen2_5_vl']
    if extras:raise ValueError('Unpinned loaded Tele plugin modules: '+str(extras))
    return dict(report,status='PASS',loaded_modules=records,
                scope='This process loaded/registered plugin; other engine processes are not directly observed')

def import_and_audit():
    inspect_installed()
    importlib.import_module('TeleOCR_vllm')
    importlib.import_module('TeleOCR_vllm.qwen2_5_vl')
    return audit_loaded()
