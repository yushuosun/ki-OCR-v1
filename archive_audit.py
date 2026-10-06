"""Audit every ZIP member against the separately approved source allowlist."""
import argparse,hashlib,json,re,stat,zipfile
from pathlib import Path,PurePosixPath
from assets import ROOT,load,sha

def audit(archive,expected=None,files_manifest_sha=None):
    expected=dict(expected if expected is not None else load(ROOT/'FILES.json'))
    expected['FILES.json']=files_manifest_sha or sha(ROOT/'FILES.json')
    forbidden={'data','models','outputs','predictions','dependencies_local','.git','.ki-ocr'}
    allowed_suffixes={'.py','.json','.md','.txt','.lock','.patch'}
    for name in expected:
        path=PurePosixPath(name)
        if forbidden.intersection(path.parts) or (path.suffix.lower() not in allowed_suffixes and name not in {'.gitignore','.gitattributes'}):
            raise ValueError('Forbidden member in external allowlist: '+name)
    directories={str(parent)+'/' for name in expected for parent in PurePosixPath(name).parents if str(parent)!='.'}
    seen=set();checked=0
    with zipfile.ZipFile(archive) as stream:
        for info in stream.infolist():
            name=info.filename;parts=PurePosixPath(name).parts
            if name in seen or name.startswith(('/', '\\')) or '\\' in name or '..' in parts:
                raise ValueError('Duplicate/unsafe ZIP member: '+name)
            seen.add(name)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('ZIP symlink rejected: '+name)
            if info.is_dir():
                if name not in directories:raise ValueError('Unlisted ZIP directory: '+name)
                continue
            if name not in expected:raise ValueError('Unlisted ZIP file: '+name)
            raw=stream.read(info)
            if hashlib.sha256(raw).hexdigest()!=expected[name]:raise ValueError('ZIP SHA mismatch: '+name)
            text=raw.decode('utf-8')
            if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)',text):
                raise ValueError('Secret pattern in ZIP member: '+name)
            checked+=1
        if {name for name in seen if not name.endswith('/')}!=set(expected):raise ValueError('ZIP missing approved files.')
    return {'status':'ZIP_ALL_MEMBERS_ALLOWLIST_SHA_PASS','source_files':checked,
            'zip_members':len(seen),'sha256':sha(archive),'secret_scan':'listed patterns only, not comprehensive DLP'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True);args=p.parse_args()
    print(json.dumps(audit(args.zip),indent=2))
