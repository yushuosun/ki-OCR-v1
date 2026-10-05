# One-page smoke configuration

This task exercises the four-stage original baseline only. Complete R3/Paddle fallback, official scoring, fresh installation and GPU execution have not been validated.

## Obtain dependencies

Use the external URLs and pinned source/file metadata in DEPENDENCIES.json and MODEL_PINS.json. The observed public Tele model revision is e92585356c0d0b7b7a65938f3da035c6593cc9a6. Four metadata files and the displayed weight SHA match the historical staged hashes; the full snapshot remains unverified. No model download or agreement acceptance is automated.

ENVIRONMENT.json records six observed framework versions. The full transitive lock and clean Linux installation are pending. The fixed upstream pyproject describes required version ranges; it is not a validated installation recipe.

## Prepare a public input

From a clean checkout and Python3.10+ environment:

~~~bash
python reproduce.py --check
python make_public_smoke.py --output data/public_smoke.png
python portable.py --doctor --config configs/portable.example.json
~~~

The one-page PNG is generated entirely by project-authored code with the standard library. It contains text and a table, needs no fonts or private assets, and is not an ODB scoring sample. Input bytes are verified before inference.

The dry run exits2 and lists missing assets until configuration is complete. Copy the example config and explicitly set tele_source, model_root, python, image, output_root and device. Relative paths resolve against the config directory, so rebase paths when moving a config. Select an explicitly configured idle device index (for example,0 on a single-GPU machine). The doctor never starts inference; GPU execution requires --run-four-stage-smoke. Output must be a new directory; existing outputs are never overwritten or resumed.

## Execute and validate

After dependencies and device configuration are ready:

~~~bash
python portable.py --config LOCAL_CONFIG.json --run-four-stage-smoke
python portable.py --verify-output OUTPUT_ROOT
~~~

These GPU commands have not been executed in the clean release environment. The launcher checks source/model hashes, framework versions, actual prompts/sampling, seed0 and context16384. It uses one cached engine and the archived baseline stage plan with table scale1.0. Sparse stages materialize only this attempt's fresh earlier artifacts.

Output validation requires nonempty Markdown and valid one-page768x768 middle JSON for NATIVE, FORMULA125, GUARD_VIEW, TABLE_VIEW and FINAL. It does not evaluate OCR accuracy or use GT to repair outputs. There is no automatic retry. The external executor must enforce the timeout and clean up spawned engine processes on failure.

## Failed experimental alternative

The baseline is the default in actual execution code and CPU assertions. --experimental-table150 explicitly selects table scale1.5 for the preserved failed alternative; all other stages are unchanged. Its once-evaluated ODB Overall was97.73879546101563, below the historical baseline98.38731063936983. configs/table150.json is marked EXPERIMENTAL_FAILED_NOT_DEFAULT. No additional public-test scale sweep is planned.

Paths can also be supplied directly without editing the example config:

~~~bash
python portable.py --doctor --tele-source /path/to/TeleOCR --model-root /path/to/model --python /path/to/venv/bin/python --image /path/to/public_smoke.png --output-root /path/to/new-output --device 0
~~~

This portable interface is separate from the archived formal resource harness; it does not alter that harness or grant execution/resource admission.
