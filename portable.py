"""Portable asset audit and explicit four-stage smoke; not the full R3 scorer."""
import argparse, ast, hashlib, importlib.metadata, json, os, shutil, struct, subprocess, sys, time, tempfile
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def resolve(cfg,key,base):
    value=cfg.get(key)
    return (base/Path(value)).resolve() if isinstance(value,str) and value else None
def inspect(config):
    cp=Path(config).resolve();c=read(cp);missing=[];deps=read(R/"DEPENDENCIES.json")
    paths={k:resolve(c,k,cp.parent) for k in ("tele_source","model_root","python","image","output_root")}
    for k,p in paths.items():
        if p is None:missing.append(k+": unset")
        elif k!="output_root" and not p.exists():missing.append(k+": not found")
        elif k in ("tele_source","model_root") and not p.is_dir():missing.append(k+": expected a directory")
        elif k in ("python","image") and not p.is_file():missing.append(k+": expected a file")
        elif k=="output_root" and p.exists():missing.append("output_root: must be a new directory")
    src=paths["tele_source"]
    if src and src.is_dir():
        for n,h in deps["tele_source_pins"].items():
            p=src/n
            if not p.is_file() or sha(p)!=h:missing.append("source mismatch: "+n)
    model=paths["model_root"]
    if model and model.is_dir():
        for name,pin in read(R/"MODEL_PINS.json")["tele"].items():
            p=model/name
            if not p.is_file():missing.append("model file missing: "+name)
            elif sha(p)!=pin["sha256"]:missing.append("frozen model file SHA mismatch: "+name)
    image=paths["image"]
    if image and image.is_file():
        raw=image.read_bytes()
        if raw[:8]!=b"\x89PNG\r\n\x1a\n":missing.append("input is not PNG")
        elif len(raw)<24:missing.append("input PNG header is truncated")
        elif struct.unpack(">II",raw[16:24])!=(768,768):missing.append("public smoke input dimensions differ")
        if sha(image)!=sha_public_input():missing.append("input differs from generated public smoke")
    gpu=c.get("device")
    if type(gpu) is not int or gpu<0:missing.append("device: require an explicit nonnegative GPU index")
    versions={}
    py=paths["python"]
    if py and py.is_file():
        script="import importlib.metadata,json\nversions={}; missing=[]\nfor k in ['vllm','torch','transformers','Pillow','lxml']:\n try: versions[k]=importlib.metadata.version(k)\n except importlib.metadata.PackageNotFoundError: missing.append(k)\nprint(json.dumps({'versions':versions,'missing':missing}))"
        try:
            p=subprocess.run([str(py),"-I","-c",script],capture_output=True,text=True,timeout=30,check=True)
            observed=json.loads(p.stdout);versions=observed["versions"]
            missing.extend("Python dependency missing: "+k for k in observed["missing"])
            for k,v in read(R/"ENVIRONMENT.json")["verified_versions"].items():
                if k in versions and versions[k]!=v:missing.append("runtime version mismatch: "+k)
        except (subprocess.SubprocessError,ValueError):missing.append("selected Python dependency inspection failed")
    ready=not missing
    return c,paths,{"status":"FOUR_STAGE_SMOKE_ASSETS_PASS" if ready else "BLOCKED","missing":missing,"runtime_versions":versions,"GPU_started":False,"full_R3_ready":False,"historical_model_equivalence":"NOT_VERIFIED","full_dependency_lock":"MISSING","four_stage_asset_check_pass":ready,"default_table_scale":1.0}
def sha_public_input():
    import importlib.util
    spec=importlib.util.spec_from_file_location("public_smoke_generator",R/"make_public_smoke.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return hashlib.sha256(mod.build()).hexdigest()
def verify_output(out):
    out=Path(out);pages=[]
    for stage in ("NATIVE","FORMULA125","GUARD_VIEW","TABLE_VIEW","FINAL"):
        md=out/stage/"markdown/public_smoke.md";middle=out/stage/"middle/public_smoke.json"
        if not md.is_file() or not md.read_text(encoding="utf-8").strip():raise ValueError("Missing/empty Markdown: "+stage)
        if not middle.is_file():raise ValueError("Missing middle JSON: "+stage)
        x=read(middle)
        if len(x.get("pdf_info",[]))!=1 or x["pdf_info"][0]["page_size"]!=[768,768]:raise ValueError("Wrong page dimensions: "+stage)
        pages.append({"stage":stage,"markdown_sha256":sha(md),"middle_sha256":sha(middle)})
    return {"status":"ONEPAGE_FOUR_STAGE_OUTPUT_PASS","pages":1,"artifacts":pages,"OCR_accuracy_evaluated":False,"full_R3_acceptance":False,"Paddle_fallback_tested":False}
def materialize(out,target,priority):
    for sub,ext in (("markdown","md"),("middle","json")):
        filename="public_smoke."+ext
        source=next((out/s/sub/filename for s in priority if (out/s/sub/filename).is_file()),None)
        if source is None:raise ValueError("Missing fresh stage output: "+filename)
        dst=out/target/sub/filename;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst)
def planned_stages(out,experimental_table150=False):
    tree=ast.parse((R/"core/entry.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="stage_plan")
    ns={"Path":Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),"frozen_stage_plan","exec"),ns)
    stages=ns["stage_plan"](out)
    assert stages[-1][1][stages[-1][1].index("--scale")+1]=="1.0"
    if experimental_table150:
        args=stages[-1][1];args[args.index("--scale")+1]="1.5"
    return stages
def run_child(config,experimental_table150=False):
    c,paths,evidence=inspect(config)
    if evidence["missing"]:raise ValueError(json.dumps(evidence))
    if sys.platform!="linux":raise ValueError("GPU smoke requires Linux")
    gpu=c["device"]
    p=subprocess.run(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
    if [int(x.strip()) for x in p.stdout.strip().split(",")]!=[0,0]:raise ValueError("Selected GPU is not idle; choose an idle device")
    out=paths["output_root"]
    if out.exists():raise ValueError("Refuse resume or overwrite")
    sys.path[:0]=[str(paths["tele_source"]),str(R/"archive")]
    import TeleOCR.config as config_module
    config_module.MAX_MODEL_LEN=16384;config_module.GPU_MEMORY_UTILIZATION=0.5
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    client=TeleOCRMODEL_SERVICE.get_model("vllm-engine",str(paths["model_root"]),None)
    from core import tele_support as support
    frozen=read(R/"configs/frozen_runtime.json")
    if support.plain(client.helper.prompts)!=frozen["expected_prompts"] or support.plain(client.helper.sampling_params)!=frozen["expected_sampling"]:raise ValueError("Actual frozen prompts/sampling mismatch")
    if client.client.model_max_length!=16384 or client.batching_mode!="stepping":raise ValueError("Actual frozen context or batching mismatch")
    if client.client.vllm_llm.llm_engine.vllm_config.model_config.seed!=0:raise ValueError("Actual seed mismatch")
    # Preserve one engine across the four archived stages.
    out.mkdir(parents=True)
    inputs=out/"INPUTS.json";inputs.write_text(json.dumps({"bound_rows":[{"id":"public_smoke.png","image":str(paths["image"])}]}))
    started=time.monotonic()
    for stage,args in planned_stages(out,experimental_table150):
        if stage=="TABLE1X":materialize(out,"GUARD_VIEW",["GUARD","FORMULA125","NATIVE"])
        filename={"NATIVE":"run_tele_img.py","FORMULA125":"stack_reread_v1.py","GUARD":"stack_guard_v1.py","TABLE1X":"stack_reread_v1.py"}[stage]
        old=sys.argv
        try:
            sys.argv=[filename,"--contract",str(inputs),"--model",str(paths["model_root"]),*args]
            code=(R/"archive"/filename).read_text()
            exec(compile(code,filename,"exec"),{"__name__":"__main__","__file__":str(R/"archive"/filename)})
        finally:sys.argv=old
    materialize(out,"TABLE_VIEW",["TABLE1X","GUARD_VIEW"])
    materialize(out,"FINAL",["TABLE_VIEW"])
    result=verify_output(out);result["wall_seconds"]=time.monotonic()-started
    (out/"RESULT.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--doctor",action="store_true",help="CPU-only asset/version/path audit; no GPU launch")
    ap.add_argument("--verify-output")
    ap.add_argument("--device",type=int,help="Explicit physical GPU index; any nonnegative index")
    for flag in ("tele-source","model-root","python","image","output-root"):
        ap.add_argument("--"+flag)
    ap.add_argument("--run-four-stage-smoke",action="store_true")
    ap.add_argument("--experimental-table150",action="store_true",help="Failed experimental candidate; never the default")
    ap.add_argument("--child",action="store_true",help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.verify_output:print(json.dumps(verify_output(a.verify_output),indent=2));return
    if a.child:run_child(a.config,a.experimental_table150);return
    if a.doctor and a.run_four_stage_smoke:ap.error("--doctor cannot launch GPU")
    cp=Path(a.config).resolve() if a.config else R/"configs/portable.example.json"
    if not cp.is_file():
        print(json.dumps({"status":"BLOCKED","missing":["config: not found"],"GPU_started":False}));raise SystemExit(2)
    try:c=read(cp)
    except (OSError,ValueError):
        print(json.dumps({"status":"BLOCKED","missing":["config: unreadable or invalid JSON"],"GPU_started":False}));raise SystemExit(2)
    for key in ("tele_source","model_root","python","image","output_root"):
        override=getattr(a,key)
        value=Path(override).resolve() if override else resolve(c,key,cp.parent)
        c[key]=str(value) if value else None
    if a.device is not None:c["device"]=a.device
    with tempfile.TemporaryDirectory(prefix="ki-ocr-config-") as tmp:
        effective=Path(tmp)/"effective.json";effective.write_text(json.dumps(c),encoding="utf-8")
        c,paths,evidence=inspect(effective)
        if not a.run_four_stage_smoke:
            print(json.dumps(evidence,indent=2));raise SystemExit(2 if evidence["missing"] else 0)
        if evidence["missing"]:print(json.dumps(evidence,indent=2));raise SystemExit(2)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(c["device"]),TELE_GPU_UTIL="0.5")
        subprocess.run([str(paths["python"]),str(Path(__file__).resolve()),"--config",str(effective),"--child"]+(["--experimental-table150"] if a.experimental_table150 else []),env=env,check=True,timeout=c["timeout_seconds"])
if __name__=="__main__":main()
