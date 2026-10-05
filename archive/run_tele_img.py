"""Company line: TeleOCR full two-step pipeline on the ORIGINAL PNG pixels (no PNG->PDF->200dpi re-render).
Layout and every crop use original pixels; Markdown built with TeleOCR's own MagicModel + union_make.
Run with PYTHONPATH=<frozen TeleOCR source>, CUDA_VISIBLE_DEVICES=<index>. Sharded, resumable."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    from PIL import Image
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    from TeleOCR.src.vlm_magic_model import MagicModel
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    from TeleOCR.tools.cut_image import cut_image_and_table
    from TeleOCR.tools.enum_class import ContentType
    from TeleOCR.tools.hash_utils import bytes_md5
    from TeleOCR.data_reader_writer import ImageDataWriter
    rows = json.loads(Path(a.contract).read_text())['bound_rows']
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    (out / 'markdown').mkdir(parents=True, exist_ok=True); (out / 'middle').mkdir(exist_ok=True)
    client = TeleOCRMODEL_SERVICE.get_model('vllm-engine', a.model, None)
    todo = [r for r in rows if not (out / 'markdown' / (Path(r['id']).stem + '.md')).exists()]
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i:i + a.chunk]
        imgs = [Image.open(r['image']).convert('RGB') for r in chunk]
        results = client.batch_two_step_extract(images=imgs)
        for r, im, blocks in zip(chunk, imgs, results):
            stem = Path(r['id']).stem
            W, H = im.size
            mm = MagicModel(blocks, W, H)
            writer = ImageDataWriter(str(out / 'images' / stem))
            md5 = bytes_md5(im.tobytes())
            for span in mm.get_all_spans():
                if span['type'] in [ContentType.IMAGE, ContentType.SEAL, ContentType.CHAR]:
                    cut_image_and_table(span, im, md5, 0, writer, scale=1.0)
            page_info = {'para_blocks': mm.get_page_blocks(), 'discarded_blocks': [], 'page_size': [W, H], 'page_idx': 0}
            md = union_make([page_info], 'images')
            (out / 'markdown' / f'{stem}.md').write_text(md, encoding='utf-8')
            (out / 'middle' / f'{stem}.json').write_text(json.dumps({'pdf_info': [page_info]}, ensure_ascii=False, default=str), encoding='utf-8')
            try:
                writer.save_all_images()
            except Exception:
                pass
        with open(out / f'progress_s{a.shard}.log', 'a') as f:
            f.write(json.dumps({'done': i + len(chunk), 'todo': len(todo)}) + '\n')


if __name__ == '__main__':
    main()
