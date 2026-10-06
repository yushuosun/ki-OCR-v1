"""CPU source audit; deliberately performs no inference or scoring."""
import argparse, ast, hashlib, json, re
from pathlib import Path
R = Path(__file__).resolve().parent
def read(name):
    return json.loads((R / name).read_text(encoding="utf-8"))
def check():
    manifest = read("FILES.json")
    actual = {p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and not ({"data","models","outputs","dependencies_local",".ki-ocr","device-locks"} & set(p.relative_to(R).parts)) and p.name not in {"LOCAL_CONFIG.json","NATIVE_DETERMINISTIC_CODE.json"}}
    assert actual == set(manifest) | {"FILES.json"}, "File allowlist mismatch"
    for name, expected in manifest.items():
        raw = (R / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, "Release hash mismatch: " + name
        text = raw.decode("utf-8")
        if name.endswith(".py"):
            ast.parse(text, filename=name)
        assert not re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)", text), "Secret pattern: " + name
        assert Path(name).suffix.lower() not in {".png", ".jpg", ".jsonl", ".pt", ".pth", ".bin", ".safetensors"}, "Prohibited artifact"
    for name, expected in read("CORE_SOURCE_PINS.json").items():
        assert manifest[name] == expected, "Frozen core mismatch: " + name
    for name, expected in read("DEPENDENCIES.json")["stage_pins"].items():
        assert manifest["archive/" + name] == expected, "Stage mismatch: " + name
    original, candidate = read("configs/original_r3.json"), read("configs/table150.json")
    assert original["table_reread_scale"] == 1.0 and candidate["table_reread_scale"] == 1.5
    ignored = {"candidate", "base_candidate", "table_reread_scale", "public_full_evaluation_count", "release_status"}
    assert {k:v for k,v in original.items() if k not in ignored} == {k:v for k,v in candidate.items() if k not in ignored}, "Unexpected scientific delta"
    assert candidate["expected_seed"] == 0 and candidate["public_pages"] == 1651
    assert not candidate["GT_used_in_inference"] and not candidate["auto_restart"]
    tree = ast.parse((R / "core/entry.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_plan")
    ns = {"Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "isolated_stage_plan", "exec"), ns)
    stages = ns["stage_plan"](Path("outputs"))
    assert [s[0] for s in stages] == ["NATIVE", "FORMULA125", "GUARD", "TABLE1X"]
    assert stages[-1][1][stages[-1][1].index("--scale")+1] == "1.0"
    assert stages[1][1][stages[1][1].index("--scale")+1] == "1.25"
    runtime = read("configs/frozen_runtime.json")
    assert runtime["expected_seed"] == 0 and runtime["max_model_len"] == 16384
    assert read("configs/portable.example.json")["table_reread_scale"] == 1.0
    assert candidate["release_status"] == "EXPERIMENTAL_FAILED_NOT_DEFAULT"
    import importlib.util
    spec = importlib.util.spec_from_file_location("portable_cpu_check", R/"portable.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    default = mod.planned_stages(Path("outputs"))
    alternative = mod.planned_stages(Path("outputs"), True)
    def scale(plan):
        args=plan[-1][1]; return args[args.index("--scale")+1]
    assert scale(default)=="1.0" and scale(alternative)=="1.5"
    assert default[:-1]==alternative[:-1]
    assert read("SCORES.json")["default_table_reread_scale"] == 1.0
    assert read("SCORES.json")["table150"]["Overall"] < read("SCORES.json")["historical_original"]["Overall"]
    # Verify the new worker changes host plumbing only, leaving the scientific
    # four-stage body equal to the frozen original after this explicit mapping.
    remap={"resource":"runtime", "lease_expires_epoch":"phase_deadline_epoch", "gpu_uuid":"device_id",
           "OWNER_LEASE_EXPIRED":"REQUEST_DEADLINE_EXPIRED",
           "REQUEST_600S_OR_OWNER_LEASE_EXPIRED":"REQUEST_600S_OR_PHASE_DEADLINE_EXPIRED",
           "PADDLE_REQUEST_600S_OR_OWNER_LEASE_EXPIRED":"PADDLE_REQUEST_600S_OR_PHASE_DEADLINE_EXPIRED"}
    class PortableHost(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in remap:
                return ast.copy_location(ast.Constant(remap[node.value]),node)
            return node
    new=ast.parse((R/"pure_worker.py").read_text(encoding="utf-8"))
    for name in ("tele_phase","paddle_phase"):
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if name=="paddle_phase":
            original.body=[s for s in original.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call)
                and any(isinstance(x,ast.Constant) and x.value=="CURRENT_PADDLE_ASSET_IDENTITY_REQUIRED" for x in s.value.args))]
        portable=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(PortableHost().visit(original)) == ast.dump(portable), "Portable scientific body changed: "+name
    assert read("DEPENDENCIES.json")["paddle_source_sha256"] == manifest["paddle_producer.py"]
    wheel = read("PADDLE_WHEEL_PIN.json")
    assert wheel['python_abi'] == 'cp310' and wheel['CUDA'] == '12.6' and wheel['version'] == '3.3.1'
    lock = set((R/'requirements-paddle.lock').read_text(encoding='utf-8').splitlines())
    assert set(wheel['CUDA_dependency_pins']) <= lock, 'Actual Paddle CUDA12 wheel dependency pin missing'
    assert not any('cu13==' in value or value.startswith(('cuda-python==','cuda-bindings==')) for value in lock), 'Stale CUDA13 Paddle lock'
    return {"status": "CPU_SOURCE_CHECK_PASS", "files_checked": len(manifest), "GPU_inference_run": False, "full_reproduction_ready": False, "external_dependencies_required": True, "secret_scan": "listed token/private-key patterns only; not comprehensive DLP"}
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", required=True)
    p.parse_args()
    print(json.dumps(check(), indent=2))
if __name__ == "__main__":
    main()
