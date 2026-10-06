"""Small synthetic CPU CDM controls; never benchmark reproduction."""
import argparse,json,os,random,subprocess,sys,time
from pathlib import Path
from assets import load,sha,write
from evaluation import renderer_check,renderer_env

def controls_pass(records):
    def rendered(item):
        value=item['metrics']
        return (not value.get('cdm_eval_error') and value.get('gt_tokens',0)>0
                and value.get('pred_tokens',0)>0 and len(item['artifacts'])>=6)
    positives=all(rendered(records[i]) and float(records[i]['metrics'].get('F1_score',0))>=.99 for i in (0,2))
    negative=rendered(records[1]) and float(records[1]['metrics'].get('F1_score',1))<float(records[0]['metrics'].get('F1_score',0))
    return bool(positives and negative)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.ki-ocr'))
    parser.add_argument('--renderer-prefix',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.absolute()
    if output.exists():raise ValueError('New synthetic smoke output required.')
    source=args.root.absolute()/'OmniDocBench'
    expected=load(Path(__file__).parent/'DEPENDENCIES.json')['scorer_revision']
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=expected:raise ValueError('Official scorer revision mismatch.')
    subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--'],check=True)
    check=renderer_check(args.root,args.renderer_prefix)
    if check['status']!='RENDERER_TOOLS_AND_CJK_FOUND_NOT_EQUIVALENCE':
        print(json.dumps(check,indent=2));raise SystemExit(2)
    env=renderer_env(args.root,args.renderer_prefix)
    os.environ.clear();os.environ.update(env)
    sys.path.insert(0,str(source))
    from src.metrics.cdm_metric import CDM
    import numpy as np
    output.mkdir(parents=True)
    start=time.monotonic();records=[]
    cases=[('latin_identity',r'E=mc^2',r'E=mc^2'),
           ('latin_negative',r'E=mc^2',r'z=1234567'),
           ('cjk_identity',r'x+\text{中文}',r'x+\text{中文}')]
    for name,reference,prediction in cases:
        random.seed(0);np.random.seed(0)
        metric=CDM(str(output/name))
        value=metric.evaluate(reference,prediction,name)
        records.append({'id':name,'synthetic':True,'metrics':value,
                        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in (output/name).rglob('*.png')}})
    passed=controls_pass(records)
    result={'status':'SYNTHETIC_CDM_CONTROLS_PASS' if passed else 'SYNTHETIC_CDM_CONTROLS_FAILED',
            'cases':records,'formula_controls':3,'source_revision':actual,'renderer':check,
            'official_renderer_equivalence':'UNVERIFIED','historical_score_reproduced':False,'benchmark_GT_read':False}
    write(output/'RESULT.json',result)
    write(output/'COST.json',{'wall_seconds':time.monotonic()-start,'GPU_started':False,'GPU_card_seconds':0})
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'SYNTHETIC_CDM_BLOCKED_OR_FAILED','type':type(error).__name__,'error':str(error)}));raise SystemExit(2)
