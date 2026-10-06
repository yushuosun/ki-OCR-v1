"""Separate official evaluator setup/configuration. Inference never imports this."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from assets import ROOT, clean_env, load, sha, write


def command(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def setup(root):
    if sys.platform != 'linux' or not (3, 10) <= sys.version_info[:2] < (3, 12):
        raise ValueError('Official pinned evaluator needs Linux Python 3.10/3.11.')
    root = root.absolute()
    source, environment = root / 'OmniDocBench', root / 'scorer_env'
    if source.exists() or environment.exists():
        raise ValueError('New scorer source/venv paths required; preserve old attempts.')
    root.mkdir(parents=True, exist_ok=True)
    deps = load(ROOT / 'DEPENDENCIES.json')
    env = clean_env(root)
    env['CUDA_VISIBLE_DEVICES'] = ''
    command(['git', 'clone', '--no-checkout', deps['scorer_url'], source], env=env)
    command(['git', '-C', source, '-c', 'core.autocrlf=false', 'checkout', '--detach', deps['scorer_revision']], env=env)
    venv.EnvBuilder(with_pip=True, symlinks=True, system_site_packages=False).create(environment)
    python = environment / 'bin/python'
    command([python, '-m', 'pip', '--isolated', 'install', '--index-url', 'https://pypi.org/simple',
             '-r', ROOT / 'requirements-scorer.lock'], env=env)
    command([python, '-m', 'pip', '--isolated', 'install', '--index-url', 'https://pypi.org/simple',
             '--no-deps', source], env=env)
    command([python, '-m', 'pip', '--isolated', 'check'], env=env)
    return {'status': 'SCORER_PYTHON_INSTALLED', 'revision': deps['scorer_revision'],
            'renderer_equivalence': 'UNVERIFIED', 'evaluation_run': False}


def prepare(root, gt, expected_gt_sha, dataset_release, predictions, output):
    import yaml
    root, gt, predictions, output = [Path(p).absolute() for p in (root, gt, predictions, output)]
    if output.exists():
        raise ValueError('New scoring output required.')
    source = root / 'OmniDocBench'
    deps = load(ROOT / 'DEPENDENCIES.json')
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != deps['scorer_revision']:
        raise ValueError('Official evaluator revision mismatch.')
    command(['git', '-C', source, 'diff', '--quiet', 'HEAD', '--'])
    if len(expected_gt_sha) != 64 or sha(gt) != expected_gt_sha.lower():
        raise ValueError('Actual GT SHA differs from explicitly supplied release.')
    if not dataset_release or dataset_release.upper() in {'UNKNOWN', 'CURRENT'}:
        raise ValueError('Explicit official dataset release name required; historical release is UNKNOWN.')
    if not predictions.is_dir() or not list(predictions.glob('*.md')):
        raise ValueError('Official Markdown prediction directory missing/empty.')
    config_source = source / 'configs/end2end.yaml'
    config = yaml.safe_load(config_source.read_text(encoding='utf-8'))
    config['end2end_eval']['dataset']['ground_truth']['data_path'] = str(gt)
    config['end2end_eval']['dataset']['prediction']['data_path'] = str(predictions)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'official_end2end.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    python = root / 'scorer_env/bin/python'
    argv = [str(python), str(source / 'pdf_validation.py'), '--config', str(config_path)]
    result = {'status': 'OFFICIAL_CONFIG_PREPARED_NOT_SCORED', 'evaluator_revision': actual,
              'original_config_sha256': sha(config_source), 'effective_config_sha256': sha(config_path),
              'dataset_release': dataset_release, 'GT_sha256': sha(gt),
              'markdown_files': len(list(predictions.glob('*.md'))), 'command': argv,
              'working_directory': str(source),
              'renderer_tools': {name: shutil.which(name) for name in ('pdflatex', 'kpsewhich', 'gs', 'magick', 'latexmlc')},
              'renderer_equivalence': 'UNVERIFIED', 'score': None}
    write(output / 'RESULT.json', result)
    return result


def renderer_env(root, prefix=None):
    root = Path(root).absolute()
    prefix = Path(prefix).absolute() if prefix else root / 'renderer'
    env = clean_env(root)
    # Never inherit an old TeX distribution/override into clean renderer validation.
    for key in list(env):
        if key.startswith(('CDM_TEXLIVE', 'OMNIDOCBENCH_TEXLIVE', 'CDM_PDFLATEX',
                           'CDM_KPSEWHICH', 'OMNIDOCBENCH_PDFLATEX', 'OMNIDOCBENCH_KPSEWHICH', 'TEXMF')):
            env.pop(key, None)
    bins = [prefix / 'portable_tex/.TinyTeX/bin/x86_64-linux', prefix / 'image/bin',
            prefix / 'tex/bin', prefix / 'env/bin', prefix / 'bin']
    env['PATH'] = os.pathsep.join(str(p) for p in bins) + os.pathsep + env.get('PATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('LD_PRELOAD', None)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(TEXMFHOME=str(prefix / 'texmf-home'), TEXMFVAR=str(prefix / 'texmf-user-var'),
               TEXMFCONFIG=str(prefix / 'texmf-user-config'))
    for directory in bins:
        if (directory / 'pdflatex').is_file():
            env['CDM_PDFLATEX'] = str(directory / 'pdflatex')
            env['CDM_KPSEWHICH'] = str(directory / 'kpsewhich')
            break
    return env


def renderer_check(root, prefix=None):
    env = renderer_env(root, prefix)
    selected = Path(prefix).absolute() if prefix else Path(root).absolute() / 'renderer'
    tools, versions, cjk, errors = {}, {}, {}, {}
    for name in ('pdflatex', 'kpsewhich', 'gs', 'magick', 'latexmlc'):
        tools[name] = shutil.which(name, path=env['PATH'])
        if tools[name]:
            try:
                if name != 'latexmlc' and not Path(tools[name]).resolve().is_relative_to(selected.resolve()):
                    raise ValueError('Required tool is outside selected public renderer prefix.')
                output = subprocess.check_output([tools[name], '--version'], env=env, text=True,
                                                 stderr=subprocess.STDOUT, timeout=20)
                versions[name] = output.splitlines()[:3]
            except Exception as error:
                errors[name] = str(error)
    if tools['kpsewhich']:
        for resource in ('article.cls', 'geometry.sty', 'booktabs.sty', 'amsmath.sty',
                         'amssymb.sty', 'upgreek.sty', 'xcolor.sty', 'multirow.sty',
                         'CJK.sty', 'c70gkai.fd'):
            check = subprocess.run([tools['kpsewhich'], resource], env=env, capture_output=True, text=True, timeout=20)
            cjk[resource] = check.stdout.strip() if check.returncode == 0 else None
            if cjk[resource] and not Path(cjk[resource]).resolve().is_relative_to(selected.resolve()):
                errors[resource] = 'Style/font resolved outside selected public renderer prefix.'
                cjk[resource] = None
    ready = all(tools[n] and n not in errors for n in ('pdflatex','kpsewhich','gs','magick')) and len(cjk) == 10 and all(cjk.values())
    return {'status': 'RENDERER_TOOLS_AND_CJK_FOUND_NOT_EQUIVALENCE' if ready else 'RENDERER_NOT_READY',
            'tools': tools, 'versions': versions, 'CJK': cjk, 'errors': errors,
            'latexml_optional_for_LaTeX_tables': True, 'synthetic_CDM_run': False,
            'official_verified_reference': {'TeX Live': '2025', 'pdfTeX':'1.40.28', 'ImageMagick':'7.1.1-47','Ghostscript':'9.55.0'},
            'official_renderer_equivalence': 'UNVERIFIED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'check', 'prepare', 'run'])
    parser.add_argument('--root', type=Path, default=Path('.ki-ocr'))
    parser.add_argument('--gt', type=Path)
    parser.add_argument('--gt-sha256')
    parser.add_argument('--dataset-release')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--renderer-prefix', type=Path)
    args = parser.parse_args()
    if args.action == 'setup':
        result = setup(args.root)
    elif args.action == 'check':
        result = renderer_check(args.root, args.renderer_prefix)
        print(json.dumps(result, indent=2))
        if result['status'] != 'RENDERER_TOOLS_AND_CJK_FOUND_NOT_EQUIVALENCE':
            raise SystemExit(2)
        return
    else:
        if not all((args.gt, args.gt_sha256, args.dataset_release, args.predictions, args.output)):
            parser.error('Require --gt, --gt-sha256, --dataset-release, --predictions and --output.')
        renderer_environment = renderer_env(args.root, args.renderer_prefix)
        os.environ.clear()
        os.environ.update(renderer_environment)
        result = prepare(args.root, args.gt, args.gt_sha256, args.dataset_release, args.predictions, args.output)
        if args.action == 'run':
            renderer_result = renderer_check(args.root, args.renderer_prefix)
            result['renderer_check'] = renderer_result
            if renderer_result['status'] != 'RENDERER_TOOLS_AND_CJK_FOUND_NOT_EQUIVALENCE':
                raise ValueError('CDM renderer tools missing; no score reported. Follow official scorer system setup.')
            smoke_argv = [str(args.root.absolute() / 'scorer_env/bin/python'),
                          str(ROOT / 'renderer_smoke.py'), '--root', str(args.root.absolute()),
                          '--output', str(args.output.absolute() / 'renderer_preflight')]
            if args.renderer_prefix:
                smoke_argv += ['--renderer-prefix', str(args.renderer_prefix.absolute())]
            with (args.output / 'renderer_preflight.log').open('x', encoding='utf-8') as stream:
                command(smoke_argv, env=renderer_environment, stdout=stream, stderr=subprocess.STDOUT)
            smoke_result = load(args.output / 'renderer_preflight/RESULT.json')
            if smoke_result['status'] != 'SYNTHETIC_CDM_CONTROLS_PASS':
                raise ValueError('Actual CDM renderer preflight failed; no official score reported.')
            result['renderer_preflight'] = {'status': smoke_result['status'],
                                          'result_sha256': sha(args.output / 'renderer_preflight/RESULT.json')}
            # Execute the official entry unchanged; GT is read only by this process.
            env = renderer_environment
            with (args.output / 'official_eval.log').open('x', encoding='utf-8') as stream:
                command(result['command'], cwd=result['working_directory'], env=env,
                        stdout=stream, stderr=subprocess.STDOUT)
            result['status'] = 'OFFICIAL_PROCESS_EXITED_ZERO_REVIEW_OUTPUT_REQUIRED'
            result['score'] = None
            write(args.output / 'RESULT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'EVALUATION_BLOCKED_OR_FAILED', 'type': type(error).__name__, 'error': str(error)}))
        raise SystemExit(2)
