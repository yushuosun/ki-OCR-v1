# ki-OCR-v1

Document parsing research code built on TeleOCR, with four-stage parsing and GT-blind conditional Paddle fallback. R3 is the internally tested pipeline version. This repository contains project code, configuration, instructions and aggregate results. It includes no original images, GT, predictions, model weights, credentials or third-party library source.

## Original R3 is the default

The default stage plan is NATIVE -> FORMULA125 -> GUARD -> TABLE1X. Formula reread scale is1.25, guard scales are1.0/0.75/1.5, and final table reread scale is **1.0**. Prompts, sampling, seed0 and context16384 are fixed. The default execution code in core/entry.py, core/tele_support.py and core/merge_formal.py is restored from the frozen original source17 version and matches its recorded hashes.

The portable launcher reads this actual baseline stage plan. Table150 is available only through the explicit --experimental-table150 flag, its experimental config and archived patch. It is a failed candidate, not the default or an improvement.

## Aggregate self-tests

| Pipeline | Overall | Status |
| --- | --- | --- |
| Original R3 (default) | 98.38731063936983 | Historical recovered composite with native16/256 reuse; not an uninterrupted fresh run |
| Table150 | 97.73879546101563 | EXPERIMENTAL failed candidate;1651 complete,0 failed |

Table150 decreased Overall by0.64851517835420 percentage points relative to the historical original. Its TEDS is0.9714836478601176, TEDS-S0.9826747513256184, CDM0.9855839064466104 and text edit distance0.024903690476259205. SCORES.json records these metrics, applicable-page denominators and the actual result SHA. No additional ODB scale sweep is planned. Original GT, the1651 denominator and historical results are unchanged.

These are self-test results, not official leaderboard acceptance. The restored baseline source has not been freshly rerun to reproduce the historical recovered score.

## CPU preparation

Python3.10+ standard library is sufficient:

~~~bash
python reproduce.py --check
python portable.py --help
python prepare_native.py --help
python make_public_smoke.py --help
~~~

The check validates source hashes, Python syntax, stage order, actual default table1.0, the explicit experimental1.5 alternative, formula/guard settings and seed/context. The CLI modules, help commands and missing-path reporting have passed in a new venv and clean directory without GPU dependencies.

**GPU one-page end-to-end and complete R3/Paddle reproduction have not passed. This is a source/preparation release, not a guarantee of successful end-to-end execution.** Full portable asset binding and a fresh Linux inference dependency installation remain unvalidated. The archived formal supervisor depends on original input/asset/resource contracts and a historical cluster-root constant; a clean checkout alone cannot run its full1651 evaluation.

See [PORTABLE_SMOKE.md](PORTABLE_SMOKE.md) for the one-page configuration and output checks. This smoke tests the four-stage parser only, not full R3 raw-event recovery, conditional Paddle fallback or official scoring.

## External assets and environment

DEPENDENCIES.json links the TeleOCR source commit9921cffe380efe4e2fa010258b3d0c3cb70bab2d, [TeleOCR model](https://huggingface.co/XingChen-AGI/TeleOCR), [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) and the [OmniDocBench scorer](https://github.com/opendatalab/OmniDocBench) revisionf133a71e9e91c3621c7ce8994200a7b394a06eb3. Obtain assets independently under their upstream terms.

MODEL_PINS.json stores hashes only. Four metadata files fetched from the fixed public Tele model revision match the historical staged hashes; its displayed weight SHA also matches. Remaining model/tokenizer files and full snapshot equivalence have not been independently fetched and verified.

Observed inference versions are vLLM0.11.0, torch2.8.0+cu128, transformers4.57.6, Paddle3.3.1, paddleocr3.7.0 and paddlex3.7.2. ENVIRONMENT.json records provenance. The transitive dependency/CUDA lock and fresh installation are incomplete; upstream version ranges are not a tested exact installation recipe.

prepare_native.py reconstructs an ignored local postprocessor bundle from independently obtained, hash-matching Tele source. It does not download code or accept agreements. Historical bundle byte identity includes an archived path and is not asserted by this reconstruction.

CPU scoring used non-Docker TeX Live2022 with an existing renderer overlay. Docker parity is unknown; formula CDM can depend on the rendering environment.

## Licensing

See [THIRD_PARTY.md](THIRD_PARTY.md). TeleOCR code-license evidence is incomplete, so its source tree and embedded native source are excluded. Third-party code/model/dataset terms are separate. The project's own license has not been selected. No agreements are accepted by these tools.
