"""DAPP S5/S6 shared logic + offline re-assembly.

Every table / interline-equation span in a DAPP middle JSON carries span['dapp'] = {candidate_name: output}:
  base      : the S3 two-step output on the decided grid
  native    : S4a table re-read from the native-grid crop
  up_grid   : S4b equation re-read from the crop upsampled to the official render density
  guard@<s> : S5 re-reads at alternative densities (only when the pre-guard selection failed validation)
Text blocks that failed validation carry block['dapp'] (same names) and block['dapp_lines_orig'].

A policy is an ordered preference list per kind plus a guard flag. Without guard the first non-None preferred
candidate is used; with guard the first *valid* preferred candidate, then the first valid guard@ candidate,
else the first preference. So one GPU pass gives every stage ablation:
  python dapp_assemble.py --middle OUT/middle --out ABL/<policy>/markdown --policy <policy>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

POLICIES = {
    'core':       {'table': ['base'], 'equation': ['base'], 'guard': False},
    'core_guard': {'table': ['base'], 'equation': ['base'], 'guard': True},
    'tab':        {'table': ['native', 'base'], 'equation': ['base'], 'guard': False},
    'tab_eq':     {'table': ['native', 'base'], 'equation': ['up_grid', 'base'], 'guard': False},
    'tab_guard':  {'table': ['native', 'base'], 'equation': ['base'], 'guard': True},
    'full':       {'table': ['native', 'base'], 'equation': ['up_grid', 'base'], 'guard': True},
}
TEXT_BLOCKS = {'text', 'title', 'list', 'ref_text', 'image_caption', 'table_caption', 'image_footnote',
               'table_footnote', 'code_caption', 'phonetic', 'aside_text', 'page_footnote'}


def crop_of(im, bbox):
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    return im.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


LEAK = re.compile(r'<nl>|<box[:>]|<\|[a-z_]+\|>|<ref>|<quad>|<[felux]cel>')


def repeated(s: str, tail: int = 256, max_period: int = 32) -> bool:
    """Decoding loop: the last `tail` chars (whitespace-collapsed) are exactly periodic with a short period.
    Legitimate repeated content (repeated cell values, \\cdots rows, leader dots followed by a number)
    does not end in a >=256-char exactly periodic tail, a degenerate decoding loop does."""
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) < tail:
        return False
    t = s[-tail:]
    return any(t[:-p] == t[p:] for p in range(1, max_period + 1))


def runaway(text, kind: str):
    """Validation failure reason, or None when the output is acceptable."""
    if text is None or not str(text).strip():
        return 'empty'
    text = str(text)
    if LEAK.search(text):
        return 'token_leak'
    if kind == 'table':
        if '<table' in text and not text.rstrip().endswith('</table>'):
            return 'truncated'
        rows = re.findall(r'<tr.*?</tr>', text, re.S)
        if not rows:
            return 'no_rows'
        cols = [len(re.findall(r'<t[dh][\s>]', r)) for r in rows]
        if max(cols) > 60 or len(rows) > 400:
            return 'table_explosion'
        if repeated(re.sub(r'<[^>]+>', ' ', text)):
            return 'repetition'
        try:
            from lxml import html as LH
            LH.fromstring(text)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            return 'unparsable_html'
        return None
    if repeated(text):
        return 'repetition'
    return None


def typed_spans(para_blocks):
    """[(span, kind, key)] for table spans (key html) and interline-equation spans (key content)."""
    out = []

    def walk(node):
        if isinstance(node, dict):
            if node.get('type') == 'table' and 'html' in node and 'bbox' in node:
                out.append((node, 'table', 'html'))
            elif node.get('type') == 'interline_equation' and 'content' in node and 'bbox' in node:
                out.append((node, 'equation', 'content'))
            for k, v in node.items():
                if k != 'dapp':
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(para_blocks)
    return out


def text_blocks(para_blocks):
    """Text-like blocks (recursing into container blocks) that have lines."""
    out = []

    def walk(node):
        if isinstance(node, dict):
            if node.get('type') in TEXT_BLOCKS and isinstance(node.get('lines'), list) and 'bbox' in node:
                out.append(node)
                return
            for k, v in node.items():
                if k not in ('dapp', 'dapp_lines_orig'):
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(para_blocks)
    return out


def block_text(blk) -> str:
    lines = blk.get('dapp_lines_orig', blk['lines'])
    return '\n'.join(''.join(str(s.get('content', '')) for s in ln.get('spans', [])) for ln in lines)


def select(cands: dict, kind: str, policy: dict, guard: bool = True):
    """Return (candidate_name, output)."""
    prefs = policy.get(kind, ['base'])
    if not (guard and policy['guard']):
        for p in prefs:
            if cands.get(p) is not None:
                return p, cands[p]
        return 'base', cands.get('base')
    for p in prefs:
        if cands.get(p) is not None and runaway(cands[p], kind) is None:
            return p, cands[p]
    for p in sorted(k for k in cands if k.startswith('guard@')):
        if runaway(cands[p], kind) is None:
            return p, cands[p]
    for p in prefs:
        if cands.get(p) is not None:
            return p, cands[p]
    return 'base', cands.get('base')


def apply_policy(page: dict, policy: dict) -> dict:
    """Set every span/block to the policy's selection in place; returns {source: count}."""
    used = Counter()
    for sp, kind, key in typed_spans(page['para_blocks']):
        if 'dapp' in sp:
            src, txt = select(sp['dapp'], kind, policy)
            sp[key] = txt
            used[f'{kind}:{src}'] += 1
    for blk in text_blocks(page['para_blocks']):
        if 'dapp' not in blk:
            continue
        src, txt = select(blk['dapp'], 'text', policy)
        if src == 'base':
            blk['lines'] = blk['dapp_lines_orig']
        else:
            blk['lines'] = [{'bbox': blk['bbox'], 'spans': [{'bbox': blk['bbox'], 'type': 'text', 'content': txt}]}]
        used[f'text:{src}'] += 1
    return dict(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--middle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--policy', required=True, choices=sorted(POLICIES))
    a = ap.parse_args()
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tot = Counter()
    for f in sorted(Path(a.middle).glob('*.json')):
        m = json.loads(f.read_text(encoding='utf-8'))
        for page in m['pdf_info']:
            tot.update(apply_policy(page, POLICIES[a.policy]))
        (out / f'{f.stem}.md').write_text(union_make(m['pdf_info'], 'images'), encoding='utf-8')
    (out.parent / f'SOURCES_{a.policy}.json').write_text(json.dumps(dict(tot), indent=1))
    print(a.policy, dict(tot))


if __name__ == '__main__':
    main()
