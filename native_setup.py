"""Pinned public OpenMP runtime in this install root; never change system libs."""
import argparse, os, subprocess, sys, time
from pathlib import Path
from assets import ROOT,clean_env,load,sha,verify_pins,write
from renderer_setup import URL,BINARY_SHA,download

def setup(root,cache=None):
    if sys.platform!='linux':raise ValueError('Native runtime requires Linux x86_64.')
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    prefix=root/'native'
    if prefix.exists():raise ValueError('Fresh native prefix required.')
    lock=load(ROOT/'NATIVE_RUNTIME_LOCK.json');packages=lock['packages']
    cache=Path(cache) if cache else None
    binary=root/'micromamba';download(URL,binary,BINARY_SHA,cache);binary.chmod(0o755)
    artifacts=[]
    for package in packages:
        artifact=root/Path(package['url']).name
        download(package['url'],artifact,package['sha256'],cache)
        artifacts.append((artifact,package))
    explicit=root/'native.explicit.lock'
    explicit.write_text('@EXPLICIT\n'+'\n'.join(artifact.as_uri()+'#'+package['md5'] for artifact,package in artifacts)+'\n')
    env=clean_env(root);env['CUDA_VISIBLE_DEVICES']=''
    started=time.monotonic()
    with (root/'native_install.log').open('x') as log:
        p=subprocess.run([str(binary),'--no-rc','-r',str(root/'native_mamba_root'),
              'create','--offline','--yes','-p',str(prefix),'--file',str(explicit)],
              env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Native runtime install failed; see native_install.log')
    count=verify_pins(prefix,lock['library_pins'])
    result={'status':'PUBLIC_NATIVE_RUNTIME_INSTALLED','library_SHA_verified':count,
            'archives':[{'sha256':sha(artifact),'package':package['name'],'version':package['version']} for artifact,package in artifacts],
            'seconds':time.monotonic()-started,'GPU_started':False,'system_changed':False}
    write(root/'NATIVE_INSTALL_RESULT.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--public-download-cache',type=Path)
    args=p.parse_args()
    print(setup(args.root,args.public_download_cache))
