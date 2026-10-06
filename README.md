# ki-OCR-v1

Frozen TeleOCR R3 document parsing with GT-blind, current-run conditional Paddle fallback. The frozen R5 standard Linux one-command online installer was run from a nonexistent root on 2026-10-06 and completed successfully: three new isolated environments, fixed public dependencies, 32 anonymous public HF model files, and actual Tele/Paddle CPU imports. Existing environments, model caches and external pip caches were not reused. The installation test did not run GPU inference or measure GPU memory/page latency; cluster-owned offline inference validation is separate evidence. The official scorer environment and synthetic CDM rendering were validated separately. Do not treat the historical score as a newly reproduced result.

This source preserves the frozen R7 inference and installation method. Its one explicit continuation was actually completed in the same initially empty installation root after a failed Tele clone, then passed one synthetic engineering page with a controlled Paddle probe. The run used a frozen local source ZIP; the public-repository-clone-to-full-install-and-GPU route has not been validated. It supports a fresh empty-root dependency installation with locally staged public models, as well as R4's explicit verified environment reuse. The frozen R3/R4 archives and scientific method remain unchanged. The standard install below targets a clean Linux host with access to HF and the public dependency repositories. Our managed cluster is a separate validation environment; its network failures do not change the public examples into private cache/service paths.

## License

Original project-authored ki-OCR-v1 code and documentation are licensed under the Apache License, Version 2.0; see [LICENSE.txt](LICENSE.txt). This grant applies only to material the respective contributors have the right to license. Third-party components, including copied or adapted third-party portions, retain their own license and attribution terms and are not relicensed by this project. See [THIRD_PARTY.md](THIRD_PARTY.md) for the recorded component terms and unresolved upstream scope.

## Install on Linux (one command, no Docker)

Use Linux x86_64 with Python 3.10 (tested 3.10.16), glibc 2.28+, Git, libGL/GLib for the frozen OpenCV wheel, and an NVIDIA driver compatible with vLLM 0.11.0 / torch 2.8.0 CUDA 12.8 and Paddle 3.3.1 CUDA 12.6. The GPU must support bfloat16. A minimum VRAM requirement has not yet been established for the unified entry.

```bash
python3.10 install.py --root .ki-ocr --download-models
```

The installer creates separate CPU, Tele and Paddle venvs with `include-system-site-packages = false`, installs the exact package versions in `requirements-*.lock`, verifies `pip check`, obtains the pinned Tele source from its public repository, and downloads ungated public HF files into `.ki-ocr/models`. It checks all recorded file SHA256s, including weights, tokenizer/processor files and the vLLM plugin. It does not use a preexisting private model cache, cluster path, internal service, token, sudo, Docker or GPU inference. It never accepts a gated-model agreement. The active command/PID and terminal exit are recorded in installation logs. Public pip downloads use timeout 300 seconds and retries 0. Failure stops with a nonzero exit and an installation log; choose a new root for a new attempt.

### One explicit continuation after a failed R5 Tele clone

Only the observed checkpoint is supported: successful `cpu_install`, `cpu_pip_check`, `cpu_environment`, `tele_install`, then failed `tele_source_clone`. Other partial-install stages stop. Supply the expected SHA256 of the original `INSTALL_RESULT.json`; the installer verifies its terminal exits, all five log hashes and exact frozen command argv, the original R5 source manifest and assets, the successful native receipt/library pins, actual same-root Python 3.10 isolated prefixes, and all 15 CPU / 179 Tele dependency versions. Tele's plugin has not been installed at this checkpoint and is deliberately not required for the dependency-only check; plugin/source/real imports still run afterward. Directory existence alone never permits skipping a step.

```bash
python3.10 install.py --root /path/to/same-incomplete-root --preloaded-models /path/to/public-models --model-manifest /path/to/OFFLINE_MODEL_MANIFEST.json --resume-install --resume-receipt-sha256 VERIFIED_ORIGINAL_FAILURE_SHA256 --resume-timeout-seconds 10800
```

The four verified successful commands are retained with `resumed_skip=true`. The continuation attempts `git -c http.version=HTTP/1.1 clone` exactly once into a new `installer-runs/resume-<id>/TeleOCR`; the original failed `TeleOCR` directory and all original logs/receipts remain untouched. No global/system Git configuration, SSL verification, proxy or network settings are changed. The new receipt and `COST.json` are in that resume directory; the original root `INSTALL_RESULT.json` remains the failed attempt. On success the newly created `runtime.json` binds the new verified Tele source, never the failed directory. The owner adapter must use this explicit new receipt and runtime source path rather than assuming the old paths. A later failed dependency install also stops; this route does not automatically retry or permit a second continuation of the same failed root.

The 10,800-second wall bound covers continuation commands, provisioning and verification checkpoints; timed-out child process groups created by this installer are terminated, with failure retained. No existing or other-owner processes are attached or terminated, and no GPU model/admission is started. The final receipt says `resumed=true`, `uninterrupted=false`, `environment_reused=true` for this same initial root's dependency envs, and `external_environment_reused=false`. “Dependencies originated in an empty root and finished by verified continuation” is supportable only with the owner's preserved original empty-root proof. It is not an uninterrupted fresh-install PASS. “First clone failed, HTTP/1.1 continuation succeeded” may be reported only after this actual clone exits 0 and all remaining installation checks pass; that real R7 continuation has now completed: the first clone failure is preserved, four completed commands were revalidated, and eleven new commands passed, including the command-level HTTP/1.1 clone. Initial installation took 934.00 seconds before failure; the continuation wrapper took 441.49 seconds (436.92 seconds inside the source installer). This is resumed same-root completion, with `resumed=true` and `uninterrupted=false`, not an uninterrupted install.

### Optional offline preloaded public models

Use this when the fixed public assets have already been downloaded on another machine that can reach HF. `OFFLINE_MODEL_MANIFEST.json` specifies the exact three public repositories, immutable revisions, 32 relative filenames, byte sizes and SHA256s. Stage only those files under `/path/to/public-models/tele`, `paddle`, and `paddle_layout`; do not copy tokens, `.cache`, private weights or old environment directories. Transferring public assets is not a new anonymous download in the destination environment.

Verify the staged directory with a standard-library-only command (no HF library import, model_info, network or token lookup):

```bash
python3.10 assets.py --verify-preloaded-models /path/to/public-models
```

For a completely fresh installation when HF is unreachable but the public dependency repositories are reachable, choose a nonexistent or completely empty installation directory and run one command:

```bash
python3.10 install.py --root /path/to/new-empty-ki-ocr --preloaded-models /path/to/public-models
```

This **models-offline, dependencies-online** route creates all three new isolated venvs, installs the same fixed public dependencies, fetches and checks the pinned Tele source, and creates its own pinned public native runtime. It performs strict offline Tele/Paddle CPU imports, then copies only the 32 verified public files into the new root and checks the copied bytes. It never calls HF model_info/download or reads a model token. Its newly generated standalone `runtime.json` points only to the new root; it cannot inherit an old runtime descriptor, native prefix or cluster path. Existing nonempty roots and `--native-prefix` are rejected. A failed dependency install or CPU import stops before model copying/runtime creation; keep the failed logs and use a different empty root for a new attempt.

HF access is unnecessary for this route, but Python 3.10 and the OS prerequisites above must already be present, and **PyPI, the public Paddle wheel address, GitHub and the public native archives must remain reachable**. It is not an entirely air-gapped installer: staging models alone cannot repair an unreachable dependency endpoint. `--public-pip-cache /path/to/public-wheel-cache` may reuse publicly downloaded wheel bytes while still creating new venvs and performing pip installs; it never reuses an installed environment. Git/native downloads remain required. Do not copy an old `public_install`, venv, runtime descriptor or private cache into the new root.

If complete isolated environments are already installed and you specifically intend to reuse them, select the following zero-network continuation instead. It checks all locked package versions, the actual installed plugin/native source, and real CPU Tele/Paddle imports, then imports only the 32 pinned public files into `.ki-ocr/models` (or verifies and reuses an already complete matching target). It never runs pip, native downloads, Git, HF requests, admission or GPU operations:

```bash
python3.10 install.py --root .ki-ocr --reuse-environments --preloaded-models /path/to/public-models
```

An optional `--model-manifest /path/to/OFFLINE_MODEL_MANIFEST.json` works in both routes and must exactly equal the shipped fixed inventory. Only the explicit reuse route allows an existing public native prefix through `--native-prefix /path/to/native-runtime`; it must pass the frozen native SHA checks. Otherwise that reuse route checks the existing runtime descriptor's native prefix or `.ki-ocr/native`. Missing environments, native libraries, source files, model entries, wrong size/SHA/revision or an outside symlink stop without network repair. A partial model destination is not merged or silently repaired. Existing installation logs/runtime remain intact; continuation records are kept in a new `installer-runs/offline-*` directory. It may create `runtime.json` only if it is absent, after all checks pass.

Offline CPU imports disable PaddleX's model-hoster connectivity probe and temporarily reject Python socket requests, HF model_info/download and token lookups. Any attempted call fails the check even if an imported library catches its exception. The receipt reports this Python API guard's observed counts; native syscall telemetry and future GPU model-loading behavior are not established by the guard.

For subsequent offline image parsing, set the process-local `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` environment variable on the inference command as well; the normal child environment inherits it. This skips PaddleX's import-time hoster probe. It does not change authentication/network settings or certify a GPU run. On a managed cluster, its original user-entry owner supplies this transient flag and all admission safeguards.

To provision models separately into an existing/new model root without touching any installed environment, use `python3.10 assets.py --import-preloaded-models /path/to/public-models --root /path/to/installation`. This verifies model files only and does not certify Python imports or inference readiness.

Acceptance is recorded separately for each installation route:

| Route / scope | Actual validation |
| --- | --- |
| Frozen R5 default `--download-models`, nonexistent root | **PASS for dependencies, anonymous public HF downloads and CPU imports.** Linux x86_64 / Python 3.10.16; the installer exited 0 after 9,042.39 wall seconds (2 h 30 min 42 s), and the wrapper including its final file verification exited 0 after 9,066.46 seconds. All three new venvs exclude system site-packages; locked package checks, installed Tele plugin/source binding, native hashes and real Tele/Paddle CPU imports passed. All 32 files (4,908,621,694 bytes) passed fixed revision/size/SHA256 checks. |
| R5 fresh preload, followed by explicit R7 continuation in the same initial root | **PASS for the resumed CPU installation and one synthetic page with a controlled Paddle probe.** The first clone failed; four completed commands were revalidated and eleven new commands passed. All 32 public models were copied and verified in that same initial root. It is `uninterrupted=false`; an uninterrupted fresh-preload install is not established. |
| R4 explicit `--reuse-environments --preloaded-models` | **PASS for existing-environment offline CPU continuation only.** This is not an empty-root install. |
| R7 same-env single-page GPU parsing | **PASS for one historically exposed synthetic engineering page and a controlled Paddle donor/merge probe.** One page returned, zero failed; 6 of the shared 32 generation items were reserved. Natural fallback did not trigger and remains **NOT VERIFIED**. Whole-directory GPU acceptance and model accuracy were not established by this smoke. |
| Whole-benchmark official evaluation / historical 98.38731 reproduction | **NOT RUN in this newly installed environment.** Exact historical GT release and renderer equivalence remain unresolved. |

The online run used the exact default command printed above, without `--reuse-environments`, `--preloaded-models` or `--public-pip-cache`. Its own freshly downloaded public wheel bytes may be shared between its new component environments. There was one installer attempt and no restart; the HF library's internal network retry count was not separately measured. An earlier 20-second Paddle wheel HEAD probe timed out; the actual pinned public wheel subsequently downloaded and installed successfully. The original probe failure remains recorded and was not rewritten as PASS. The complete install log SHA256 is `e0a656aad1628fa555e5bea54ff17802790192cb7155f72fe5c901741111e6d5`. Keep `INSTALL_RESULT.json`, `MODEL_DOWNLOADS.json`, the real CPU import reports, `pyvenv.cfg` files and `runtime.json` with that log. This scope certifies CPU imports and fixed assets, not GPU model construction, natural fallback, minimum VRAM, performance or a benchmark score. A successful cluster run, offline continuation or old prepared environment cannot replace an independent fresh online install.

Actual R4 CPU evidence: all 32 staged files (4,908,621,694 bytes) passed revision/size/SHA checks. The verified continuation ran on the existing isolated public-download Linux environments and passed actual Tele/Paddle CPU imports; their Python socket/HF-request/token guard counts were all zero. It reused matching models without copying or reinstalling them, taking 240.17 wall seconds. This is evidence from that local Linux environment, not a successful cluster/GPU run. Linux regression coverage totals 68 controls (the original 50 plus 18 offline controls); Windows cannot run the two Linux symlink fixtures and does not request additional privileges for them.

R5 adds eight synthetic CPU controls for the fresh preloaded route: all three new venvs and a new standalone runtime, an existing empty root, nonempty-root rejection, external-native-prefix rejection, bad models before installation side effects, dependency failure, attempted offline import network access, and preservation of the default online-HF download route. All 76 controls passed on Linux Python 3.10.16 in 2.02 seconds; Windows passed 74 and skipped the two Linux symlink fixtures. The Linux test runner used a newly created CPU-test-only venv with the frozen Pillow 11.3.0, not an old inference venv. The earlier bare-Python attempt failed two old input controls because Pillow was absent; that failed log is retained. These tests do not install the full Tele/Paddle dependencies. Keep the actual empty-root install log, `INSTALL_RESULT.json`, the two strict import reports, `PRELOADED_MODEL_IMPORT.json` (32 copied files), new-root `pyvenv.cfg` files and `runtime.json` when that real run is performed. Do not write PASS until they exist and have been checked; archive a failed run without substituting evidence from an old environment.

The locks pin versions. The actual official Paddle CUDA12.6 / cuDNN9.5.1 cp310 wheel is additionally pinned by URL/SHA256 in `PADDLE_WHEEL_PIN.json`; other Python wheels are not hash-locked. The old Paddle dependency snapshot contained CUDA13 libraries and was corrected to the 15 exact CUDA12 package versions required by this actual wheel METADATA. This changes dependency preparation, not the frozen inference scripts or model settings. The project's Apache-2.0 grant covers its original material only. Tele's repository-wide code-license scope and other third-party terms remain as recorded in [THIRD_PARTY.md](THIRD_PARTY.md); these tools do not grant additional rights to upstream software.

The installer obtains the pinned public `libgomp` and `_openmp_mutex` conda-forge archives using a fixed public micromamba binary into `.ki-ocr/native`. It verifies archive and library SHA256s and selects that library directory only for its own child processes; it does not change system libraries. This repairs the actual missing `libgomp.so.1` found in the fresh Paddle import. Real CPU import preflights run after installation; `pip check` alone does not prove native imports work. The standard all-in-one online command completed from a nonexistent root without interruption on 2026-10-06, including both real CPU import checks. The separate preloaded-model route has a verified R7 same-root continuation after clone failure; this does not establish uninterrupted fresh installation.

Tele uses the pinned **top-level TeleOCR 0.1.0 project installed with `--no-deps`**, together with the 179-version lock and Transformers 4.57.6. The separately packaged plugin's Transformers 4.57.1 installation route is not used or mixed in. `UPSTREAM_PLUGIN_PINS.json` binds both public plugin files; runtime checks bind their actual installed files, unique effective registration and loaded source/loader paths. Identical build-generated source egg-info is deduplicated; conflicting metadata or another plugin install is rejected. Other engine processes are not directly observed by this parent-process source audit.

Public assets are fixed at:

| Asset | Revision |
| --- | --- |
| Tele source | `9921cffe380efe4e2fa010258b3d0c3cb70bab2d` |
| `XingChen-AGI/TeleOCR` | `e92585356c0d0b7b7a65938f3da035c6593cc9a6` |
| `PaddlePaddle/PaddleOCR-VL-1.6` | `c5630abae1d940eafe0697512a0325494b02ab42` |
| `PaddlePaddle/PP-DocLayoutV3` | `7b48a7566925fa464281f930c58eee04fe2c862a` |

## Parse a page or the entire image directory (one command)

Select an idle physical GPU you own explicitly. On a managed cluster, the existing owner admission adapter must be used; the public CLI does not admit or register cluster jobs. Our owner bridge keeps the real owner lease, identity, locks, selected GPU and generation budget separate from the inference algorithm. Each model child retains the original live UID/boot, lease expiry, physical UUID and logical CUDA mapping checks. It rejects GPU0/7 in that cluster path and does not reuse old identities. `core/entry.py` and the original cluster safeguards remain unchanged.

```bash
.ki-ocr/cpu_env/bin/python inference.py run --input /path/page.png --output outputs/page --runtime .ki-ocr/runtime.json --device 1 --paddle-probe
```

```bash
.ki-ocr/cpu_env/bin/python inference.py run --input /path/OmniDocBench/images --output outputs/odb --runtime .ki-ocr/runtime.json --device 1
```

Input directories are flat PNG/JPEG directories. Pages are sorted by filename; duplicate filename stems and malformed images fail explicitly. The output must be new. The official end-to-end evaluator consumes `outputs/odb/markdown/<input-stem>.md`, including HTML tables and LaTeX math. No filename rewriting or GT lookup is used.

A CPU-only asset/input check uses `check` in place of `run` and a separate new output path. It checks isolated environment metadata, every frozen package version, actual installed plugin files and native/model/source hashes. Its result explicitly says `METADATA_AND_FILES_ONLY_NOT_INFERENCE_READY`; it does not construct models or certify GPU execution. The installer separately runs real CPU import preflights.

`RESULT.json` records page states, failures, natural fallback count, finish-reason observations and method identity. `COST.json` records phase times, wall time per input page, selected-device sampled peak MiB and component memory reports when available. Every failed/missing page retains an output file and its denominator. A failed fallback retains the primary output, reports an error and causes a nonzero CLI exit. Raw answers, intermediate results and predictions remain local to the output directory. Automatic retries are disabled.

`--paddle-probe` requests one real same-input Paddle donor only when natural fallback is absent, then tests an explicitly synthetic empty-primary merge. It does not alter the final prediction or claim natural-failure coverage. A failed child or empty merge fails the probe, sets `overall_success=false` and `INFERENCE_WITH_ERRORS`, and exits nonzero while retaining the primary output.

## Frozen method and historical result

The default is **NATIVE → FORMULA125 → GUARD → TABLE1X**: seed 0, context 16384, native vLLM stepping, original prompts/sampling, formula scale 1.25, guard scales 1.0/0.75/1.5 and final table scale 1.0. Recovery policy is `historical_latest_abnormal`. Deterministic native recovery precedes a Paddle donor; donor replacement requires the original unique geometric match and exact Markdown interval rules. This is the TeleOCR + conditional Paddle R3 method, separate from the company API pipeline.

The historical **98.38731063936983** Overall came from recovered/composed 1651 outputs with native16/256 reuse. It was not an uninterrupted new-environment run. The precise official GT release name is unknown; the historical evaluator commit was `f133a71e9e91c3621c7ce8994200a7b394a06eb3`. Official renderer equivalence was not established. The new entry has not reproduced that score; it preserves the frozen method without further ODB tuning. [SCORES.json](SCORES.json) retains the historical values and failed Table150 candidate. Table150 is available only through the older explicitly experimental tools and is never selected by the new entry.

The new entry explicitly locks seed 0 and context 16384, with `enforce_eager=False` by default; `--eager` selects and records True. The SHA-matched historical front/back/recovery contracts record context 16384 and the Tele model, but do not record seed or eager: their actual seed/eager remain **UNKNOWN**. Older independent R6 component smokes used eager True. Neither those smokes nor this new default prove the historical run configuration.

R7 same-env synthetic one-page observation: Tele phase 210.03 seconds, controlled Paddle probe 9.06 seconds, worker wall time 225.24 seconds and total reserved GPU cost 225.23 card-seconds. These include phase startup/loading and do not establish steady-state page latency. Selected-device samples every approximately two seconds reached 39,959 MiB during Tele and 8,383 MiB during Paddle; these are sampled device usage, not tensor allocated memory or guaranteed minimum VRAM. The final Markdown file was produced (96 bytes). All tracked own model children exited; physical GPU availability was not independently verified. Requests completed and the Tele process exited 0; a shutdown-time EngineCore **ERROR** was recorded, cause UNKNOWN. This engineering smoke does not prove OCR accuracy, natural fallback coverage, whole-ODB performance or 98.38731 reproduction. The published source adds only the portable historical evidence-root API described below; that documentation/root-parameterization revision has not been rerun on GPU. Historical independent component evidence: one Tele page took 4.589 s after load; its parent torch memory counters were zero and cannot measure vLLM engine memory. A separate Paddle page took 1.820 s after load, with 3,199,085,824 allocated bytes and 7,486,318,592 reserved bytes. These observations are not new-entry performance or a minimum VRAM guarantee.

## Official evaluation (separate from inference)

Install the pinned official evaluator Python environment (`requirements-scorer.lock` freezes the dependencies observed in a clean Linux installation):

```bash
python3.10 evaluation.py setup --root .ki-ocr
```

Install the public renderer in your own directory, with no Docker or sudo (patched Python 3.10.12+ is needed for safe archive extraction):

```bash
python3.10 renderer_setup.py --root .ki-ocr/renderer
```

This installs pinned ImageMagick 7.1.1-47 / Ghostscript 10.06 conda-forge archives and TinyTeX 2026.10 with SHA256/SHA512-checked CJK/font/`was` archives. No system TeX, authentication or ImageMagick security policy is changed. A changed/missing public archive stops installation. Check tools, styles and CJK fonts and run actual synthetic CPU CDM controls:

```bash
.ki-ocr/scorer_env/bin/python renderer_smoke.py --root .ki-ocr --output outputs/renderer-smoke
```

Actual tests with the pinned official scorer passed: `E=mc^2` versus itself F1=1.000 (5 tokens); `x+\text{中文}` versus itself F1=1.000 (4 tokens); `E=mc^2` versus `z=1234567` F1=0.286. All cases produced real PNGs; these three synthetic controls took 36.42 wall seconds on CPU. The final source-installed fixed renderer and isolated font paths repeated the controls successfully in 23.90 seconds. Initial tests without `upgreek.sty` silently returned zero; `was` supplies that style. `evaluation.py run` now requires the same real positive/negative preflight before starting the official evaluation. These are renderer controls, not benchmark scores.

This is a new renderer. The official README recommends TeX Live 2025 / pdfTeX 1.40.28, ImageMagick 7.1.1-47 and Ghostscript 9.55.0. Our actual TeX Live 2026 / pdfTeX 1.40.29 and Ghostscript 10.06 have **UNVERIFIED official/historical equivalence**. LaTeXML for optional LaTeX-table conversion is not installed or validated; inference emits HTML tables.

Provide your independently obtained official GT file, its verified SHA256 and exact release name. The historical release name is not guessed or defaulted. Prepare the original official metric/matching config with only the two data paths changed:

```bash
.ki-ocr/cpu_env/bin/python evaluation.py prepare --root .ki-ocr --gt /path/official.json --gt-sha256 VERIFIED_SHA256 --dataset-release EXACT_OFFICIAL_RELEASE --predictions outputs/odb/markdown --output outputs/evaluation
```

The generated `RESULT.json` prints the actual official command:

```bash
.ki-ocr/scorer_env/bin/python .ki-ocr/OmniDocBench/pdf_validation.py --config /absolute/path/outputs/evaluation/official_end2end.yaml
```

Run it with working directory `.ki-ocr/OmniDocBench` and the renderer's bin directories in PATH, or use `evaluation.py run` with the same arguments and a new scoring output directory. The wrapper selects its own renderer and runs the unchanged official entry after real CDM preflight. The pinned official evaluator installed in a separate clean Linux venv and its `--help` command ran. A whole-benchmark evaluation has **not run** in this new environment. Missing tools, styles, CJK fonts or failed CDM controls block evaluation. No guarantee that this renderer reproduces historical CDM scores is made. GT, original scores and original submission bundles are read-only.

## Historical audit helper portability

The historical formal-acceptance helper now requires `acceptance(contract, root=AUTHORIZED_ROOT)` with an explicit caller-authorized, existing canonical absolute directory. Missing/relative/file/symlink or whole-filesystem roots fail closed; the root is never inferred from contract data. All prior approval, SHA, evidence-size, path-boundary, owner/resource and lease checks remain. The legacy `core/entry.py` live `FRESH_FORMAL_RUN` path still calls this helper without an explicit root and is intentionally disabled by fail-closed rejection in this release until its original owner explicitly binds an authorized root. This release supports the documented `inference.py` and `install.py` flow; it does not claim compatibility for every historical live entry. The public inference path uses `pure_worker.contract_rows` and `inference.checked_rows`, rather than this legacy formal-acceptance default. Four-stage inference bodies, model settings, prompts, thresholds and third-party pins remain unchanged. This is a release portability change, separate from the preserved R7 source ZIP and its actual GPU evidence.

## CPU regression controls

```bash
.ki-ocr/cpu_env/bin/python -m unittest discover -s tests -v
python3.10 reproduce.py --check
```

The original 50 CPU controls cover unchanged healthy text, independent two-page directory output, whole-page fallback, failed-slot exact replacement, stale/cross-page raw rejection, bad/missing donor retention, input/native-bundle freezes, missing/changed package versions, a same-version wrong CUDA wheel, inherited model caches, negative owner admission, changed UID/boot, expired child lease, physical/logical UUID mismatches, silent/ineffective CDM controls, successful/failed/empty controlled Paddle probes, installed plugin conflicts, and extra/duplicate archive members. They use synthetic inputs and a clearly synthetic identity conversion fixture; they do not test real model accuracy or GPU execution. `reproduce.py --check` verifies the source allowlist, pins and frozen stages. `archive_audit.py --zip SOURCE.zip` independently checks every ZIP member against the external source allowlist and SHA256s, including directories and symlink rejection.

The current release has 94 CPU controls: all passed on Linux; Windows passed 92 and skipped two Linux symlink fixtures. R7 adds 13 receipt/continuation controls and this release adds five explicit-root/boundary controls. These are synthetic CPU checks, with no dependency installation or GPU model execution.

`PUBLIC_ASSET_MANIFEST.json` is included in the source allowlist and archive, with public model32/plugin2/native2 identities. The external delivery `RESULT.json` binds the actual source commit and archive SHA; these are kept outside the archive to avoid self-hash cycles.

This repository contains project source, configuration, hashes and aggregate historical scores only. Original images, GT, prediction bodies, weights, secrets and internal runtime artifacts are excluded. Public distribution must preserve this source-only boundary and the separate third-party terms.
