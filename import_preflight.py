"""Actual CPU imports and source binding, with no visible GPU or model creation."""
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from assets import ROOT,clean_env,load,verify_pins,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('tele','paddle'),required=True)
    p.add_argument('--native-prefix',type=Path,required=True);p.add_argument('--tele-source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strict-offline',action='store_true',help='Reject Python network/HF/token operations during CPU imports')
    args=p.parse_args()
    prefix=args.native_prefix.absolute()
    verify_pins(prefix,load(ROOT/'NATIVE_RUNTIME_LOCK.json')['library_pins'])
    # Loader search paths must be set before Python loads any native extension.
    if os.environ.get('KI_OCR_CPU_IMPORT_PREFLIGHT')!='1':
        env=clean_env(prefix.parent,prefix/'lib');env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',KI_OCR_CPU_IMPORT_PREFLIGHT='1')
        if args.strict_offline:
            env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
        os.execve(sys.executable,[sys.executable,'-I',str(ROOT/'import_preflight.py'),*sys.argv[1:]],env)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU import requires no visible GPU.')
    if os.environ.get('LD_LIBRARY_PATH')!=str(prefix/'lib'):raise ValueError('Explicit public native prefix required.')
    print('CPU_IMPORT_START '+args.kind,flush=True)
    result={'kind':args.kind,'GPU_started':False,'model_constructed':False}
    guard=None
    if args.strict_offline:
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
        from offline_import_guard import OfflineImportGuard
        guard=OfflineImportGuard().install()
    if args.kind=='tele':
        if args.tele_source is None:raise ValueError('Pinned public Tele source required.')
        sys.path[:0]=[str(args.tele_source.absolute()),str(ROOT/'core')]
        import torch,vllm,transformers
        print('BASE_TELE_IMPORTED',flush=True)
        _=vllm.LLM
        print('LLM_CLASS_IMPORTED_NO_CONSTRUCTOR',flush=True)
        from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
        from plugin_audit import import_and_audit
        result['plugin']=import_and_audit()
        from source_audit import audit
        result['tele_source']=audit(args.tele_source,load(ROOT/'DEPENDENCIES.json')['tele_source_pins'],sys.modules)
        if result['tele_source']['status']!='PASS':raise ValueError('Actual Tele source import audit rejected.')
        result['versions']={'torch':torch.__version__,'vllm':vllm.__version__,'transformers':transformers.__version__}
    else:
        import paddle,cv2
        print('PADDLE_CORE_OPENCV_IMPORTED',flush=True)
        from paddleocr import PaddleOCRVL
        result['versions']={'paddle':paddle.__version__,'opencv':cv2.__version__}
    if guard:
        guard.require_no_attempts()
        result.update(strict_offline=True,offline_guard=guard.report())
        guard.close()
    result['status']='CPU_IMPORT_PREFLIGHT_PASS_NOT_GPU_ACCEPTANCE'
    write(args.output,result);print(result['status'],flush=True)

if __name__=='__main__':main()
