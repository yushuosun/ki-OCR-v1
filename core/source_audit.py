"""Strict frozen-source audit, including manifest-backed namespace packages."""
import hashlib
import importlib.machinery as machinery
from importlib import _bootstrap_external
from pathlib import Path

NamespaceLoader = getattr(machinery, 'NamespaceLoader', _bootstrap_external._NamespaceLoader)


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root, pins, modules):
    root=Path(root).resolve(); records=[]; errors=[]
    # All frozen files, including critical modules, retain hash verification.
    for relative, expected in pins.items():
        p=(root/relative).resolve()
        try:
            require(p.is_relative_to(root) and sha(p)==expected, 'FROZEN_SOURCE_PIN_CHANGED')
        except (ValueError,OSError) as e: errors.append({'file':relative,'reason':str(e)})
    for name,module in sorted(list(modules.items())):
        if name!='TeleOCR' and not name.startswith('TeleOCR.'): continue
        spec=getattr(module,'__spec__',None); loader=getattr(module,'__loader__',None)
        file=getattr(module,'__file__',None)
        record={'name':name,'file':file,'spec_name':getattr(spec,'name',None),
                'spec_origin':getattr(spec,'origin',None),
                'loader_type':type(loader).__module__+'.'+type(loader).__qualname__,
                'search_locations':list(getattr(spec,'submodule_search_locations',None) or [])}
        try:
            require(spec is not None and spec.name==name and spec.loader is loader, 'MODULE_SPEC_LOADER_IDENTITY_UNKNOWN')
            if file is None:
                relative=name.replace('.','/'); expected=(root/relative).resolve()
                require(type(loader) is NamespaceLoader and spec.origin is None and not spec.has_location,
                        'NO_FILE_MODULE_NOT_VERIFIED_NAMESPACE')
                locations=[Path(p).resolve() for p in record['search_locations']]
                require(expected.is_relative_to(root) and expected.is_dir() and locations==[expected],
                        'NAMESPACE_PATH_OUTSIDE_FROZEN_ROOT_OR_MULTIPLE')
                require('__init__.py' not in [p.name for p in expected.iterdir()] and
                        relative+'/__init__.py' not in pins and
                        any(p.startswith(relative+'/') for p in pins), 'NAMESPACE_NOT_MANIFEST_BACKED')
                require([Path(p).resolve() for p in getattr(module,'__path__',[])]==locations,
                        'NAMESPACE_MODULE_PATH_CHANGED')
                record['state']='VERIFIED_NAMESPACE_NO_SOURCE_FILE'
            else:
                p=Path(file).resolve()
                require(p.is_relative_to(root),'TELE_MODULE_FROM_WRONG_SOURCE_ROOT')
                relative=p.relative_to(root).as_posix()
                require(relative in pins and sha(p)==pins[relative],'LOADED_TELE_MODULE_PIN_CHANGED')
                require(type(loader) is machinery.SourceFileLoader and spec.has_location and
                        Path(spec.origin).resolve()==p and Path(loader.path).resolve()==p and loader.name==name,
                        'SOURCE_LOADER_ORIGIN_MISMATCH')
                record['state']='VERIFIED_PINNED_SOURCE'
        except (ValueError,OSError,TypeError) as e:
            record['state']='REJECTED'; record['reason']=str(e); errors.append(record.copy())
        records.append(record)
    return {'status':'PASS' if not errors else 'REJECTED','frozen_files':len(pins),
            'modules':records,'errors':errors}
