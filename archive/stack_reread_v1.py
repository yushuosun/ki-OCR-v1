"""Stack DAPP per-type density re-reads on an existing full-page run (e.g. the F0 fine-tuned model).

Reads <run>/middle/<stem>.json (run_tele_img layout, page_size == original PNG size), re-extracts the chosen
kinds from original-PNG crops resampled by --scale with --model (may differ from the model that produced the
layout), keeps a re-read only when it passes dapp_assemble.runaway validation, writes Markdown via union_make.
Pages without a re-readable span are written byte-identical to the source Markdown, so sparse scoring is exact.
Run with PYTHONPATH=<frozen TeleOCR source>:<dir of this file>:<lxml dir>, CUDA_VISIBLE_DEVICES=<index>.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import PIL.Image
from PIL import Image

from dapp_assemble import crop_of, runaway, typed_spans

PIL.Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='dir with markdown/ and middle/')
    ap.add_argument('--contract', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kinds', default='equation')
    ap.add_argument('--scale', type=float, default=200 / 72)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args()
    kinds = set(a.kinds.split(','))
    import TeleOCR.config as CONFIG
    if os.environ.get('TELE_GPU_UTIL'):
        CONFIG.GPU_MEMORY_UTILIZATION = float(os.environ['TELE_GPU_UTIL'])
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    run, out = Path(a.run), Path(a.out)
    (out / 'markdown').mkdir(parents=True, exist_ok=True); (out / 'middle').mkdir(exist_ok=True)
    rows = json.loads(Path(a.contract).read_text())['bound_rows']
    todo, stats = [], {'pages': 0, 'pages_with_spans': 0, 'spans': 0, 'replaced': 0, 'rejected_invalid': 0}
    for r in rows:
        stem = Path(r['id']).stem
        if (out / 'markdown' / f'{stem}.md').exists():
            continue
        m = json.loads((run / 'middle' / f'{stem}.json').read_text(encoding='utf-8'))
        spans = [(sp, k, key) for sp, k, key in typed_spans(m['pdf_info'][0]['para_blocks']) if k in kinds]
        if not spans:
            shutil.copyfile(run / 'markdown' / f'{stem}.md', out / 'markdown' / f'{stem}.md')
            stats['pages'] += 1
            continue
        todo.append((r, stem, m, spans))
    client = TeleOCRMODEL_SERVICE.get_model('vllm-engine', a.model, None)
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i:i + a.chunk]
        crops, types, refs = [], [], []
        for r, stem, m, spans in chunk:
            page = m['pdf_info'][0]
            im = Image.open(r['image']).convert('RGB')
            sx, sy = im.width / page['page_size'][0], im.height / page['page_size'][1]
            for sp, k, key in spans:
                x0, y0, x1, y1 = sp['bbox']
                c = crop_of(im, (x0 * sx, y0 * sy, x1 * sx, y1 * sy))
                if a.scale != 1.0:
                    c = c.resize((max(1, round(c.width * a.scale)), max(1, round(c.height * a.scale))), Image.Resampling.LANCZOS)
                crops.append(c); types.append(k); refs.append((sp, k, key))
        res = client.batch_content_extract(crops, types) if crops else []
        for (sp, k, key), txt in zip(refs, res):
            stats['spans'] += 1
            if txt is None or runaway(txt, k):
                stats['rejected_invalid'] += 1
                continue
            sp[key + '_stack_orig'] = sp.get(key); sp[key] = txt; stats['replaced'] += 1
        for r, stem, m, spans in chunk:
            (out / 'markdown' / f'{stem}.md').write_text(union_make(m['pdf_info'], 'images'), encoding='utf-8')
            (out / 'middle' / f'{stem}.json').write_text(json.dumps(m, ensure_ascii=False, default=str), encoding='utf-8')
            stats['pages'] += 1; stats['pages_with_spans'] += 1
        (out / 'STATS.json').write_text(json.dumps(stats, indent=1))
    (out / 'STATS.json').write_text(json.dumps(stats, indent=1))
    (out / 'EXIT.json').write_text(json.dumps({'exit': 0}))


if __name__ == '__main__':
    main()
