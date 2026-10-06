"""Pinned public user-directory CDM resources; no sudo or policy changes."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
from assets import ROOT,clean_env,load,sha,write
URL='https://github.com/mamba-org/micromamba-releases/releases/download/2.3.2-0/micromamba-linux-64'
BINARY_SHA='ffc3cb8d52d4d6b354bdbb979c407719c485392b74e462cbd50811aa88e58f85'

def download(url,dest,expected,cache=None):
    cached=Path(cache)/dest.name if cache else None
    if cached and cached.is_file() and sha(cached)==expected:
        shutil.copyfile(cached,dest)
    else:
        request=urllib.request.Request(url,headers={'User-Agent':'ki-ocr-public-installer'})
        with urllib.request.urlopen(request,timeout=60) as source,dest.open('xb') as target:
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                target.write(chunk)
    if sha(dest)!=expected:raise ValueError('Public renderer archive SHA changed: '+dest.name)

def setup(root,components=('image','tex'),cache=None):
    if sys.platform!='linux':raise ValueError('Renderer installation requires Linux x86_64.')
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    locks=load(ROOT/'RENDERER_LOCKS.json');env=clean_env(root);env['CUDA_VISIBLE_DEVICES']=''
    previous=load(root/'RENDERER_INSTALL_RESULT.json') if (root/'RENDERER_INSTALL_RESULT.json').is_file() else {}
    records=previous.get('commands',[])
    previous_components=previous.get('components',[])
    started_total=time.monotonic()
    def command(argv,name):
        started=time.monotonic()
        with (root/(name+'_install.log')).open('x') as log:
            process=subprocess.Popen([str(x) for x in argv],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(root/'RENDERER_INSTALL_RESULT.json',{'status':'INSTALLING','commands':records,'active_pid':process.pid,'active_command':name,'GPU_started':False})
            process.wait()
        record={'command':name,'exit':process.returncode,'seconds':time.monotonic()-started,'log_sha256':sha(root/(name+'_install.log'))}
        records.append(record)
        write(root/'RENDERER_INSTALL_RESULT.json',{'commands':records,'GPU_started':False})
        if process.returncode:raise RuntimeError('Public renderer command failed: '+name)
    if 'image' in components:
        if (root/'image').exists():raise ValueError('Fresh image prefix required.')
        binary=root/'micromamba';download(URL,binary,BINARY_SHA,cache);binary.chmod(0o755)
        packages=locks['components']['image'];explicit=root/'image.explicit.lock'
        explicit.write_text('@EXPLICIT\n'+'\n'.join(p['url']+'#'+p['md5'] for p in packages)+'\n')
        command([binary,'--no-rc','-r',root/'mamba-root','create','-p',root/'image','--yes','--file',explicit],'image')
        # Full SHA verification of downloaded public archives and actual linked package metadata.
        for package in packages:
            artifact=root/'mamba-root/pkgs'/Path(package['url']).name
            if not artifact.is_file() or sha(artifact)!=package['sha256']:
                raise ValueError('Public renderer package SHA mismatch/missing: '+package['name'])
            metadata=list((root/'image/conda-meta').glob(package['name']+'-'+package['version']+'-*.json'))
            if len(metadata)!=1:raise ValueError('Actual renderer package metadata missing/ambiguous: '+package['name'])
            actual=load(metadata[0])
            if actual['version']!=package['version'] or actual['build']!=package['build_string']:
                raise ValueError('Actual renderer package version/build differs: '+package['name'])
        records[-1]['package_archive_SHA_verified']=len(packages)
    if 'tex' in components:
        tex_root=root/'portable_tex'
        if tex_root.exists():raise ValueError('Fresh TeX prefix required.')
        if not hasattr(tarfile,'data_filter'):raise ValueError('Use patched Python 3.10.12+ or 3.11.4+ for safe TeX extraction.')
        item=locks['tinytex'];archive=root/'TinyTeX-v2026.10.tar.xz'
        download(item['url'],archive,item['sha256'],cache)
        tex_root.mkdir()
        with tarfile.open(archive) as stream:stream.extractall(tex_root,filter='data')
        tex=tex_root/'.TinyTeX';package_root=root/'tex_archives';package_root.mkdir()
        archives=[]
        for package in locks['tex_packages']:
            target=package_root/(package['name']+'.tar.xz')
            download(package['url'],target,package['sha256'],Path(cache)/'tex_archives' if cache else None)
            if hashlib.sha512(target.read_bytes()).hexdigest()!=package['containerchecksum']:raise ValueError('TeX archive SHA512 mismatch: '+package['name'])
            archives.append(target)
        env.update(TEXMFHOME=str(root/'texmf-home'),TEXMFVAR=str(root/'texmf-user-var'),TEXMFCONFIG=str(root/'texmf-user-config'))
        # Documented --file mode receives the complete frozen dependency set.
        command([tex/'bin/x86_64-linux/tlmgr','install','--file',*archives],'tex_local_packages')
        command([tex/'bin/x86_64-linux/mktexlsr'],'tex_filename_database')
        records[-1]['tinytex_SHA_verified']=item['sha256'];records[-1]['extra_archives_SHA_verified']=len(archives)
    result={'status':'PINNED_PUBLIC_RENDERER_RESOURCES_INSTALLED_NOT_CDM_VERIFIED','commands':records,
            'components':sorted(set(previous_components)|set(components)),
            'this_invocation_wall_seconds':time.monotonic()-started_total,
            'GPU_started':False,'official_renderer_equivalence':'UNVERIFIED','ImageMagick_security_policy_changed':False,
            'synthetic_CDM_run':False,'root':str(root)}
    write(root/'RENDERER_INSTALL_RESULT.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.ki-ocr/renderer'))
    parser.add_argument('--components',choices=('all','image','tex'),default='all')
    parser.add_argument('--public-download-cache',type=Path,help='Optional public SHA-matched renderer archives only.')
    args=parser.parse_args()
    env=clean_env(args.root);os.environ.clear();os.environ.update(env)
    components=('image','tex') if args.components=='all' else (args.components,)
    try:print(json.dumps(setup(args.root,components,args.public_download_cache),indent=2))
    except Exception as error:
        failure={'status':'RENDERER_INSTALL_FAILED','type':type(error).__name__,'error':str(error),'GPU_started':False}
        if args.root.is_dir():write(args.root/'RENDERER_FAILURE.json',failure)
        print(json.dumps(failure));raise SystemExit(2)

if __name__=='__main__':main()
