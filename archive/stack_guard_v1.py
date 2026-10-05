"""DAPP S5 guard stacked on an existing run: re-read only blocks whose output fails dapp_assemble.runaway.

Per page the middle JSON comes from --run/middle (if present) else --fallback/middle; Markdown for untouched pages
is copied byte-identically from --run/markdown. Text blocks / tables / equations that fail validation are
re-extracted from original-PNG crops at --scales (in order) with --model; the first valid output replaces the
block (text blocks become one text span). Nothing valid -> original kept. Writes GUARD_LOG.jsonl.
Run with PYTHONPATH=<frozen TeleOCR source>:<dir of this file>:<lxml dir>, CUDA_VISIBLE_DEVICES=<index>.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import PIL.Image
from PIL import Image

from dapp_assemble import block_text, crop_of, runaway, text_blocks, typed_spans

PIL.Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--fallback', required=True)
    ap.add_argument('--contract', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scales', default='1.0,0.75,1.5')
    a = ap.parse_args()
    scales = [float(s) for s in a.scales.split(',')]
    run, fb, out = Path(a.run), Path(a.fallback), Path(a.out)
    (out / 'markdown').mkdir(parents=True, exist_ok=True); (out / 'middle').mkdir(exist_ok=True)
    jobs = []  # (row, stem, middle, [(obj, kind, key|None)])
    for r in json.loads(Path(a.contract).read_text())['bound_rows']:
        stem = Path(r['id']).stem
        mp = run / 'middle' / f'{stem}.json'
        m = json.loads((mp if mp.exists() else fb / 'middle' / f'{stem}.json').read_text(encoding='utf-8'))
        page = m['pdf_info'][0]
        bad = [(sp, k, key) for sp, k, key in typed_spans(page['para_blocks']) if runaway(sp.get(key), k)]
        bad += [(b, 'text', None) for b in text_blocks(page['para_blocks'])
                if runaway(block_text(b), 'text') not in (None, 'empty')]
        if bad:
            jobs.append((r, stem, m, bad))
        else:
            shutil.copyfile(run / 'markdown' / f'{stem}.md', out / 'markdown' / f'{stem}.md')
    import TeleOCR.config as CONFIG
    if os.environ.get('TELE_GPU_UTIL'):
        CONFIG.GPU_MEMORY_UTILIZATION = float(os.environ['TELE_GPU_UTIL'])
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    client = TeleOCRMODEL_SERVICE.get_model('vllm-engine', a.model, None)
    log = open(out / 'GUARD_LOG.jsonl', 'w')
    for r, stem, m, bad in jobs:
        page = m['pdf_info'][0]
        im = Image.open(r['image']).convert('RGB')
        sx, sy = im.width / page['page_size'][0], im.height / page['page_size'][1]
        for obj, kind, key in bad:
            x0, y0, x1, y1 = obj['bbox']
            base = crop_of(im, (x0 * sx, y0 * sy, x1 * sx, y1 * sy))
            before = obj.get(key) if key else block_text(obj)
            rec = {'page': stem, 'kind': kind, 'reason': runaway(before, kind), 'accepted_scale': None}
            eq_mult = 200 / 72 if kind == 'equation' else 1.0
            for s in scales:
                f = s * eq_mult
                c = base if abs(f - 1) < 1e-6 else base.resize((max(1, round(base.width * f)), max(1, round(base.height * f))), Image.Resampling.LANCZOS)
                txt = client.batch_content_extract([c], [kind])[0]
                if txt and runaway(txt, kind) is None:
                    if key:
                        obj[key + '_guard_orig'] = obj.get(key); obj[key] = txt
                    else:
                        obj['guard_lines_orig'] = obj['lines']
                        obj['lines'] = [{'bbox': obj['bbox'], 'spans': [{'bbox': obj['bbox'], 'type': 'text', 'content': txt}]}]
                    rec['accepted_scale'] = s
                    break
            log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()
        (out / 'markdown' / f'{stem}.md').write_text(union_make(m['pdf_info'], 'images'), encoding='utf-8')
        (out / 'middle' / f'{stem}.json').write_text(json.dumps(m, ensure_ascii=False, default=str), encoding='utf-8')
    (out / 'EXIT.json').write_text(json.dumps({'exit': 0, 'pages_guarded': len(jobs)}))


if __name__ == '__main__':
    main()
