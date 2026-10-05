"""Build an ignored local native bundle from independently obtained pinned source."""
import argparse, hashlib, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tele-source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / "DEPENDENCIES.json").read_text())["native_postprocessor_file_sha256"]
    source = a.tele_source / "TeleOCR" / "vlm_utils" / "post_process"
    files = {}
    for name, expected in pins.items():
        raw = (source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise SystemExit("Pinned source mismatch: " + name)
        files[name] = raw.decode("utf-8")
    if a.output.exists():
        raise SystemExit("Refusing to overwrite existing bundle")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"source": str(source), "files": files}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verified_files": len(files), "historical_bundle_byte_identity_claimed": False}))
if __name__ == "__main__":
    main()
