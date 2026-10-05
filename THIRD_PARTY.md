# External dependencies, checked 2026-10-05

| Dependency | Primary source | Observed status | Distribution here |
| --- | --- | --- | --- |
| TeleOCR model card | https://huggingface.co/XingChen-AGI/TeleOCR/raw/main/README.md | Model card declares Apache-2.0; not proof of the exact old snapshot or every code file's terms | Link and hashes; no weights |
| TeleOCR code | https://github.com/caipeng328/TeleOCR | Repository and pyproject inspected; current top-level LICENSE URL returned404. Frozen code commit is9921cffe380efe4e2fa010258b3d0c3cb70bab2d. Model-card terms are not automatically asserted for the full code tree | Link and per-file hashes; no third-party source tree or embedded native code |
| OmniDocBench scorer | https://github.com/opendatalab/OmniDocBench/blob/main/LICENSE | Apache License2.0 file observed; frozen scorer revision f133a71e9e91c3621c7ce8994200a7b394a06eb3 | Link/revision; no scorer or dataset copied |
| PaddleOCR code | https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE | Apache License2.0 file observed | Link; no library/model copied |

The included stage adapters, supervisor and recovery policy are project research code. Their own public-release license has not been selected here. No broad claim that all third-party code is cleared is made. Dataset and model distribution terms must be checked at their original sources for the exact version actually obtained; no agreement is accepted by these preparation tools.

No original images, GT, company evaluation data, prediction bodies, credentials, model weights or complete experiment logs are included. Full benchmark prediction Markdown is a separate potential review artifact and has not been approved for this repository upload.
