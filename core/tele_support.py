"""Fresh fixed20 Tele B0 worker; original pinned four-stage scripts, no cache.

No vllm/torch/model import at module import, --help, --validate-only or --mock.
Real execution requires parent GPU admission and an externally enforced deadline.
Raw prompts/answers stay in the owned PRIVATE files, never in stdout metadata.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import sys
import threading
import time
import traceback


SCHEMA = "E3_TELE_FRESH_FIXED20_B0_WORKER_R1"
STAGES = ("NATIVE", "FORMULA125", "GUARD", "TABLE1X")
UNKNOWN = "UNKNOWN"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def plain(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return {k: plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    except TypeError:
        return str(value)


def image_identity(image):
    if isinstance(image, list):
        return [image_identity(x) for x in image]
    if isinstance(image, bytes):
        return {"bytes": len(image), "encoded_sha256": hashlib.sha256(image).hexdigest()}
    if hasattr(image, "tobytes"):
        return {"size": list(image.size), "mode": image.mode,
                "RGB_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    return {"type": str(type(image)), "identity": UNKNOWN}


class BudgetStop(RuntimeError):
    pass


class RequestBudget:
    """Whole batches are reserved before calling the engine; never split/cap shrink."""
    def __init__(self, deadline, maximum, batch_timeout):
        if not math.isfinite(deadline) or not 0 < maximum <= 2048:
            raise ValueError("INVALID_FROZEN_REQUEST_OR_DEADLINE_BUDGET")
        if batch_timeout != 600:
            raise ValueError("FROZEN_GENERATION_BATCH_TIMEOUT_MUST_BE_600")
        self.deadline = deadline
        self.maximum = maximum
        self.batch_timeout = batch_timeout
        self.submitted = 0

    def remaining(self):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise BudgetStop("GLOBAL_CARD_WALL_DEADLINE_EXHAUSTED")
        return remaining

    def reserve(self, count):
        self.remaining()
        if count < 0 or self.submitted + count > self.maximum:
            raise BudgetStop("WHOLE_BATCH_WOULD_EXCEED_FROZEN_TOTAL_REQUEST_BUDGET")
        self.submitted += count
        return min(self.batch_timeout, self.remaining())


def input_rows(contract, contract_path):
    p = Path(contract["input_contract"])
    if not p.is_absolute():
        p = Path(contract_path).resolve().parent / p
    if sha(p) != contract["input_contract_sha256"]:
        raise RuntimeError("INPUT_CONTRACT_BYTES_CHANGED")
    rows = read_json(p)["input_rows"]
    if len(rows) != 20 or len({r["page_id"] for r in rows}) != 20:
        raise RuntimeError("FIXED20_INPUT_CARDINALITY_OR_DUPLICATE")
    if len({r["doc_id"] for r in rows}) != 10:
        raise RuntimeError("FIXED10_DOCUMENT_CARDINALITY")
    for r in rows:
        if Path(r["page_id"]).name != r["page_id"] or Path(r["page_id"]).stem != r["page_id"]:
            raise RuntimeError("UNSAFE_PAGE_ID")
    return p, rows


def pin_map(contract):
    result = {}
    for pin in contract["external_pins"]:
        path = str(Path(pin["path"]).resolve())
        if path in result and result[path] != pin["sha256"]:
            raise RuntimeError("CONFLICTING_EXTERNAL_SOURCE_PIN")
        result[path] = pin["sha256"]
    return result


def require_pin(path, pins):
    p = str(Path(path).resolve())
    if p not in pins or sha(p) != pins[p]:
        raise RuntimeError("LOADED_SOURCE_OR_MODEL_PIN_CHANGED: " + p)
    return pins[p]


def validate(contract, contract_path, verify_files=True):
    p, rows = input_rows(contract, contract_path)
    tele = contract["tele"]
    cfg = tele["config"]
    expected = {"max_model_len": 16384, "gpu_memory_utilization": 0.5,
                "enforce_eager": True, "compilation_level": 0, "cudagraph_mode": "NONE"}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError("FROZEN_TELE_CONFIG_MISMATCH: " + key)
    if contract["budget"]["generation_batch_timeout_seconds"] != 600:
        raise RuntimeError("GENERATION_TIMEOUT_CONTRACT_MISMATCH")
    pins = pin_map(contract)
    archive = Path(tele["archive_root"])
    required_scripts = [archive / n for n in ("run_tele_img.py", "stack_reread_v1.py", "stack_guard_v1.py", "dapp_assemble.py")]
    for script in required_scripts:
        if str(script.resolve()) not in pins:
            raise RuntimeError("UNPINNED_ORIGINAL_STAGE_CODE: " + str(script))
    model_pins = [p for p in pins if Path(tele["model"]).resolve() in Path(p).parents]
    if len(model_pins) != 13:
        raise RuntimeError("EXACT13_TELE_MODEL_PROCESSOR_PINS_REQUIRED")
    source_pins = [p for p in pins if Path(tele["source_root"]).resolve() in Path(p).parents]
    if len(source_pins) < 55:
        raise RuntimeError("FULL_FROZEN_TELE_SOURCE_PINS_REQUIRED")
    file_checks = []
    if verify_files:
        from PIL import Image
        pdf_checked = set()
        for r in rows:
            if sha(r["image"]) != r["image_sha256"]:
                raise RuntimeError("ORIGINAL_INPUT_PNG_BYTES_CHANGED: " + r["page_id"])
            with Image.open(r["image"]) as im:
                im.load()
                if im.mode != "RGB" or list(im.size) != r["size"]:
                    raise RuntimeError("ORIGINAL_INPUT_RGB_OR_DIMENSIONS_CHANGED")
                if image_identity(im)["RGB_pixel_sha256"] != r["RGB_pixel_sha256"]:
                    raise RuntimeError("ORIGINAL_INPUT_PIXELS_CHANGED")
            if r["pdf_path"] not in pdf_checked:
                if sha(r["pdf_path"]) != r["pdf_sha256"]:
                    raise RuntimeError("ORIGINAL_INPUT_PDF_BYTES_CHANGED")
                pdf_checked.add(r["pdf_path"])
            file_checks.append({"page_id": r["page_id"], "image_sha256": r["image_sha256"], "status": "PASS"})
        for source in pins:
            require_pin(source, pins)
    return {"schema": SCHEMA, "status": "PASS", "input_contract_sha256": sha(p),
            "pages": 20, "documents": 10, "verified_images": len(file_checks),
            "verified_external_pins": len(pins) if verify_files else 0,
            "stage_invocations": [
                {"stage": "NATIVE", "script": "run_tele_img.py", "chunk": 16, "fresh_full_two_step": True},
                {"stage": "FORMULA125", "script": "stack_reread_v1.py", "scale": 1.25, "chunk": 32},
                {"stage": "GUARD", "script": "stack_guard_v1.py", "scales": [1.0, 0.75, 1.5], "equation_factor": 200 / 72},
                {"stage": "TABLE1X", "script": "stack_reread_v1.py", "scale": 1.0, "chunk": 32}],
            "batches": [16, 4], "GPU_started": False, "model_imported": False,
            "reference_or_model_answers_read": False}, rows, pins


class Worker:
    def __init__(self, contract, contract_path, job, deadline, max_requests):
        self.contract = contract
        self.contract_path = Path(contract_path)
        self.run_contract_sha256 = sha(self.contract_path)
        self.worker_sha256 = sha(__file__)
        self.job = Path(job)
        self.tele = contract["tele"]
        self.rows = []
        self.pins = {}
        allowed = min(max_requests, int(contract["budget"]["max_total_requests"]))
        self.budget = RequestBudget(deadline, allowed, int(contract["budget"]["generation_batch_timeout_seconds"]))
        self.state = {"schema": SCHEMA, "utc": utc(), "stage": "VALIDATION", "batch": None,
                      "run_contract_sha256": self.run_contract_sha256,
                      "worker_sha256": self.worker_sha256,
                      "calls": 0, "submitted_requests": 0, "completed_requests": 0,
                      "completed_pages": [], "input_pages": 20, "input_documents": 10,
                      "model_loading_started": False, "GPU_started": False,
                      "reference_provided_to_model": False, "cache_reuse": False}
        self.bindings = []
        self.page_ids = []
        self.crop_candidates = {}
        self.started = time.monotonic()
        self.local = threading.local()
        self.preparation_events = []
        self.preparation_lock = threading.Lock()

    def save_state(self):
        self.state.update(utc=utc(), submitted_requests=self.budget.submitted,
                          wall_seconds=time.monotonic() - self.started)
        put(self.job / "STATE.json", self.state)

    def arm_deadline(self):
        signal.setitimer(signal.ITIMER_REAL, self.budget.remaining())

    def install_observers(self, client, llm_class):
        from PIL import Image
        import numpy as np
        self.client = client
        old_layout = client.helper.batch_prepare_for_layout
        old_extract = client.helper.batch_prepare_for_extract
        old_single_extract = client.helper.prepare_for_extract
        old_crop = Image.Image.crop
        original_generate = llm_class.generate

        def observed_crop(image, box=None):
            crop = old_crop(image, box)
            active = getattr(self.local, "crop_events", None)
            if active is not None:
                active.append({"integer_or_float_crop_box_actual": [v.item() if hasattr(v, "item") else v for v in box] if box is not None else None,
                               "parent": image_identity(image), "crop_before_rotation_resize": image_identity(crop)})
            return crop

        def observed_single(image, blocks, *args, **kwargs):
            self.local.crop_events = []
            try:
                result = old_single_extract(image, blocks, *args, **kwargs)
                with self.preparation_lock:
                    self.preparation_events.append({"parent_object_id": id(image), "crop_events": self.local.crop_events})
                return result
            finally:
                self.local.crop_events = None

        def observed_layout(executor, images, *args, **kwargs):
            result = old_layout(executor, images, *args, **kwargs)
            if len(images) != len(result) or len(images) != len(self.page_ids):
                raise RuntimeError("LAYOUT_HELPER_CARDINALITY_MISMATCH")
            self.bindings = []
            for index, (parent, prepared) in enumerate(zip(images, result)):
                row = self.by_id[self.page_ids[index]]
                binding = {"page_id": row["page_id"], "input_image_sha256": row["image_sha256"],
                           "image_index": index, "kind": "layout", "bbox": [0, 0, *parent.size],
                           "parent_image": image_identity(parent), "prepared_crop": image_identity(prepared)}
                self.bindings.append(binding)
                append(self.job / "PROCESSOR_INPUT_CHAIN.jsonl", {"utc": utc(), "stage": self.state["stage"],
                       "batch": self.state["batch"], "binding": binding, "helper": "batch_prepare_for_layout"})
            return result

        def observed_extract(executor, images, blocks_list, *args, **kwargs):
            # PIL crop is observed inside each actual helper call, including executor threads.
            self.preparation_events = []
            result = old_extract(executor, images, blocks_list, *args, **kwargs)
            self.bindings = []
            for index, (parent, blocks, prepared) in enumerate(zip(images, blocks_list, result)):
                crops, prompts, params, indices = prepared
                if not len(crops) == len(prompts) == len(params) == len(indices):
                    raise RuntimeError("EXTRACT_HELPER_CARDINALITY_MISMATCH")
                width, height = parent.size
                if width * height > self.config.MAX_PIXELS:
                    scale = (self.config.MAX_PIXELS / (width * height)) ** 0.5
                    width, height = max(1, round(width * scale)), max(1, round(height * scale))
                for crop, block_index in zip(crops, indices):
                    block = blocks[block_index]
                    pts = np.array(block.bbox, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] *= width
                    pts[:, 1] *= height
                    integer_points = pts.astype(np.int32).tolist()
                    rectangle = [*integer_points[0], *integer_points[1]] if len(integer_points) == 2 else UNKNOWN
                    if self.state["stage"] == "NATIVE":
                        row = self.by_id[self.page_ids[index]]
                        # MagicModel uses Python float multiplication then int on the original page.
                        # The helper crop independently uses np.float32 then int32, which may differ by 1px.
                        normalized = list(block.bbox)
                        magic_candidate = [int(float(v) * (parent.width if i % 2 == 0 else parent.height))
                                           for i, v in enumerate(normalized)] if len(normalized) == 4 else UNKNOWN
                        base = {"page_id": row["page_id"], "input_image_sha256": row["image_sha256"],
                                "kind": str(block.type), "bbox": magic_candidate,
                                "bbox_basis": "MAGICMODEL_PYTHON_FLOAT_INT_ON_ORIGINAL_PAGE_CANDIDATE_REQUIRES_CURRENT_POSTPROCESS_ASSOCIATION"}
                    else:
                        key = (str(block.type), image_identity(parent).get("RGB_pixel_sha256"))
                        candidates = self.crop_candidates.get(key, [])
                        base = candidates[0].copy() if len(candidates) == 1 else {
                            "page_id": UNKNOWN, "input_image_sha256": UNKNOWN, "kind": str(block.type),
                            "bbox": UNKNOWN, "source_identity": "UNKNOWN_AMBIGUOUS_OR_UNMATCHED",
                            "candidates": candidates}
                    binding = {**base, "image_index": index, "block_index": block_index,
                               "helper_normalized_bbox": plain(block.bbox),
                               "helper_actual_integer_points": integer_points,
                               "helper_crop_rectangle": rectangle, "angle": plain(block.angle),
                               "parent_image": image_identity(parent), "prepared_crop": image_identity(crop),
                               "geometry_basis": "ACTUAL_HELPER_BLOCK_BBOX_AND_FROZEN_FLOAT32_INT32_CONVERSION",
                               "magic_model_bbox_association": UNKNOWN}
                    event_matches = [entry["crop_events"] for entry in self.preparation_events if entry["parent_object_id"] == id(parent)]
                    events = event_matches[0] if len(event_matches) == 1 else UNKNOWN
                    if isinstance(events, list) and rectangle != UNKNOWN:
                        exact = [e for e in events if e["integer_or_float_crop_box_actual"] == rectangle]
                        binding["actual_PIL_crop_box_verified"] = bool(exact)
                        binding["actual_PIL_crop_events_for_bbox"] = exact
                    else:
                        binding["actual_PIL_crop_box_verified"] = UNKNOWN
                    self.bindings.append(binding)
                    append(self.job / "PROCESSOR_INPUT_CHAIN.jsonl", {"utc": utc(), "stage": self.state["stage"],
                           "batch": self.state["batch"], "binding": binding,
                           "helper": "batch_prepare_for_extract", "executor_is_none": executor is None,
                           "actual_PIL_crop_events_available": isinstance(events, list),
                           "actual_PIL_crop_events_this_parent": events})
            return result

        def observed_generate(llm, *args, **kwargs):
            prompts = args[0] if args else kwargs.get("prompts", [])
            prompts = prompts if isinstance(prompts, list) else [prompts]
            params = args[1] if len(args) > 1 else kwargs.get("sampling_params")
            param_list = params if isinstance(params, list) else [params] * len(prompts)
            timeout = self.budget.reserve(len(prompts))
            call = self.state["calls"]
            self.state["calls"] += 1
            self.save_state()
            input_meta = []
            for index, prompt in enumerate(prompts):
                text = prompt.get("prompt", "") if isinstance(prompt, dict) else str(prompt)
                text_ids = client.client.tokenizer.encode(text, add_special_tokens=False)
                image = prompt.get("multi_modal_data", {}).get("image") if isinstance(prompt, dict) else None
                sampling = plain(param_list[index])
                if getattr(param_list[index], "max_tokens", None) != 16384:
                    raise RuntimeError("FROZEN_MAX_TOKENS_CHANGED_NO_CAP_SHRINK_ALLOWED")
                extra = getattr(param_list[index], "extra_args", None)
                if not isinstance(extra, dict) or extra.get("no_repeat_ngram_size") != 100:
                    raise RuntimeError("FROZEN_NO_REPEAT_NGRAM_CHANGED")
                binding = self.bindings[index] if len(self.bindings) == len(prompts) else {
                    "page_id": UNKNOWN, "input_image_sha256": UNKNOWN, "kind": UNKNOWN,
                    "bbox": UNKNOWN, "source_identity": "UNKNOWN_GENERATE_BINDING_CARDINALITY"}
                meta = {"index": index, "binding": binding, "actual_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "actual_text_token_count": len(text_ids), "actual_text_token_ids_sha256": object_sha(text_ids),
                        "text_tokens_are_whole_context": False, "actual_prompt_image": image_identity(image),
                        "sampling": sampling, "output_cap_requested": 16384}
                input_meta.append(meta)
                append(self.job / "PRIVATE_PROMPT_REQUESTS.jsonl", {"utc": utc(), "call": call, "stage": self.state["stage"],
                       **meta, "actual_prompt_text": text, "actual_text_token_ids": text_ids})
            append(self.job / "CALL_STARTS.jsonl", {"utc": utc(), "stage": self.state["stage"],
                   "batch": self.state["batch"], "call": call, "requests": len(prompts),
                   "inputs": input_meta, "timeout_seconds_effective": timeout,
                   "global_deadline_epoch": self.budget.deadline, "whole_batch_reserved": True})
            put(self.job / "REQUEST_INFLIGHT.json", {"utc": utc(), "call": call, "requests": len(prompts), "status": "RUNNING"})
            signal.setitimer(signal.ITIMER_REAL, timeout)
            t = time.monotonic()
            try:
                outputs = original_generate(llm, *args, **kwargs)
            finally:
                # Preserve outputs that returned at the wall boundary before reporting budget stop.
                # The parent supervisor independently enforces the same absolute hard deadline.
                signal.setitimer(signal.ITIMER_REAL, 0)
                remaining = self.budget.deadline - time.time()
                if remaining > 0:
                    signal.setitimer(signal.ITIMER_REAL, remaining)
            if len(outputs) != len(prompts):
                raise RuntimeError("GENERATION_CARDINALITY_MISMATCH")
            for index, output in enumerate(outputs):
                self.state["completed_requests"] += 1
                meta = input_meta[index]
                choices = []
                for choice in output.outputs:
                    choices.append({"text": choice.text, "raw_text_sha256": hashlib.sha256(choice.text.encode()).hexdigest(),
                                    "token_ids": list(choice.token_ids), "output_tokens": len(choice.token_ids),
                                    "finish_reason": choice.finish_reason, "stop_reason": plain(choice.stop_reason)})
                first = choices[0] if choices else {}
                raw = {"utc": utc(), "stage": self.state["stage"], "batch": self.state["batch"], "call": call,
                       "run_contract_sha256": self.run_contract_sha256,
                       "page_id": meta["binding"].get("page_id", UNKNOWN), "binding": meta["binding"],
                       "source": {"worker_sha256": self.worker_sha256, "stage_code": self.active_script,
                                  "stage_code_sha256": require_pin(self.active_script, self.pins),
                                  "path": self.active_script, "script_sha256": require_pin(self.active_script, self.pins),
                                  "run_contract_sha256": self.run_contract_sha256},
                       "source_script_sha256": require_pin(self.active_script, self.pins),
                       "request_id": output.request_id, "batch_index": index, "outputs": choices,
                       "raw_text": first.get("text"), "finish_reason": first.get("finish_reason"),
                       "stop_reason": first.get("stop_reason"), "finished": getattr(output, "finished", UNKNOWN),
                       "actual_prompt_sha256": meta["actual_prompt_sha256"], "sampling": meta["sampling"],
                       "actual_text_token_count": meta["actual_text_token_count"],
                       "actual_engine_prompt_token_count": len(output.prompt_token_ids or []),
                       "actual_engine_prompt_token_ids_sha256": object_sha(output.prompt_token_ids),
                       "context_remaining_from_engine_prompt_tokens": max(0, 16384 - len(output.prompt_token_ids)) if output.prompt_token_ids is not None else UNKNOWN,
                       "context_count_basis": "vllm.RequestOutput.prompt_token_ids_actual",
                       "engine_minus_plain_text_prompt_tokens": len(output.prompt_token_ids or []) - meta["actual_text_token_count"],
                       "multi_modal_placeholders": plain(getattr(output, "multi_modal_placeholders", UNKNOWN)),
                       "vision_token_count": UNKNOWN,
                       "image_tokens_and_context": "ACTUAL_ENGINE_PROMPT_TOKEN_IDS_RECORDED_DISTINCT_FROM_PURE_TEXT",
                       "output_cap_requested": 16384,
                       "processor_source": "PINNED_NATIVE_MODEL_PROCESSOR_AND_ACTUAL_PREPARED_IMAGE"}
                append(self.job / "PRIVATE_RAW_REQUESTS.jsonl", raw)
            append(self.job / "CALL_ENDS.jsonl", {"utc": utc(), "call": call, "stage": self.state["stage"],
                   "requests": len(outputs), "wall_seconds": time.monotonic() - t,
                   "output_tokens": sum(len(c.token_ids) for o in outputs for c in o.outputs)})
            put(self.job / "REQUEST_INFLIGHT.json", {"utc": utc(), "call": call, "status": "COMPLETE"})
            self.save_state()
            self.budget.remaining()
            return outputs

        Image.Image.crop = observed_crop
        client.helper.prepare_for_extract = observed_single
        client.helper.batch_prepare_for_layout = observed_layout
        client.helper.batch_prepare_for_extract = observed_extract
        llm_class.generate = observed_generate

    def capture_crop_candidates(self, directory, rows, stage):
        from PIL import Image
        from dapp_assemble import typed_spans, text_blocks, block_text, runaway, crop_of
        self.crop_candidates = {}
        for row in rows:
            stem = row["page_id"]
            middle = directory / ("NATIVE" if stage == "FORMULA125" else "GUARD_VIEW") / "middle" / (stem + ".json")
            if stage == "GUARD":
                middle = next(directory / s / "middle" / (stem + ".json") for s in ("FORMULA125", "NATIVE")
                              if (directory / s / "middle" / (stem + ".json")).exists())
            page = read_json(middle)["pdf_info"][0]
            spans = typed_spans(page["para_blocks"])
            if stage in ("FORMULA125", "TABLE1X"):
                kind = "equation" if stage == "FORMULA125" else "table"
                targets = [(obj, k, n) for n, (obj, k, _) in enumerate(spans) if k == kind]
            else:
                targets = [(obj, k, n) for n, (obj, k, key) in enumerate(spans) if runaway(obj.get(key), k)]
                targets += [(obj, "text", n) for n, obj in enumerate(text_blocks(page["para_blocks"]))
                            if runaway(block_text(obj), "text") not in (None, "empty")]
            if not targets:
                continue
            with Image.open(row["image"]) as image:
                image = image.convert("RGB")
                sx, sy = image.width / page["page_size"][0], image.height / page["page_size"][1]
                for obj, kind, index in targets:
                    x0, y0, x1, y1 = obj["bbox"]
                    actual = [int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)]
                    base = crop_of(image, (x0 * sx, y0 * sy, x1 * sx, y1 * sy))
                    scales = [1.25] if stage == "FORMULA125" else [1.0] if stage == "TABLE1X" else [1.0, 0.75, 1.5]
                    for scale in scales:
                        factor = scale * (200 / 72 if stage == "GUARD" and kind == "equation" else 1.0)
                        crop = base if abs(factor - 1) < 1e-6 else base.resize(
                            (max(1, round(base.width * factor)), max(1, round(base.height * factor))), Image.Resampling.LANCZOS)
                        binding = {"page_id": row["page_id"], "input_image_sha256": row["image_sha256"],
                                   "kind": kind, "bbox": obj["bbox"], "source_middle_sha256": sha(middle),
                                   "source_span_or_block_index": index, "original_input_crop_integer_bbox": actual,
                                   "reread_scale": scale, "equation_multiplier": 200 / 72 if stage == "GUARD" and kind == "equation" else 1.0,
                                   "source_identity": "UNIQUE_CURRENT_SOURCE_CROP_PIXEL_MATCH"}
                        key = (kind, image_identity(crop)["RGB_pixel_sha256"])
                        self.crop_candidates.setdefault(key, []).append(binding)

    def invoke_stage(self, directory, rows, name, script, args):
        self.budget.remaining()
        if sha(self.contract_path) != self.run_contract_sha256:
            raise RuntimeError("RUN_CONTRACT_CHANGED_DURING_EXECUTION")
        if sha(__file__) != self.worker_sha256:
            raise RuntimeError("ACTUAL_WORKER_CODE_CHANGED_DURING_EXECUTION")
        for row in rows:
            if sha(row["image"]) != row["image_sha256"]:
                raise RuntimeError("CURRENT_STAGE_INPUT_PNG_CHANGED: " + row["page_id"])
        self.state["stage"] = name
        self.bindings = []
        self.active_script = str(Path(self.tele["archive_root"]) / script)
        code_hash = require_pin(self.active_script, self.pins)
        if name != "NATIVE":
            self.capture_crop_candidates(directory, rows, name)
        self.save_state()
        append(self.job / "STAGES.jsonl", {"utc": utc(), "batch": self.state["batch"], "stage": name,
               "status": "STARTED", "source": self.active_script, "source_sha256": code_hash,
               "requests_so_far": self.budget.submitted})
        old_argv = sys.argv
        try:
            sys.argv = [self.active_script, "--contract", str(directory / "INPUTS.json"), "--model", self.tele["model"], *args]
            runpy.run_path(self.active_script, run_name="__main__")
        except BaseException:
            append(self.job / "STAGES.jsonl", {"utc": utc(), "batch": self.state["batch"], "stage": name,
                   "status": "FAILED_OR_BUDGET_STOP", "requests_so_far": self.budget.submitted})
            raise
        finally:
            sys.argv = old_argv
        require_pin(self.active_script, self.pins)
        append(self.job / "STAGES.jsonl", {"utc": utc(), "batch": self.state["batch"], "stage": name,
               "status": "COMPLETE", "source_sha256": code_hash, "requests_so_far": self.budget.submitted})

    def check_loaded_modules(self):
        observed = []
        for name, module in list(sys.modules.items()):
            if name == "TeleOCR" or name.startswith("TeleOCR.") or name == "dapp_assemble":
                path = getattr(module, "__file__", None)
                if not path:
                    continue
                if Path(path).suffix == ".pyc":
                    path = __import__("importlib.util", fromlist=["source_from_cache"]).source_from_cache(path)
                observed.append({"module": name, "path": path, "sha256": require_pin(path, self.pins)})
        put(self.job / "LOADED_MODULE_PINS.json", {"utc": utc(), "modules": observed})

    def execute(self):
        if os.environ.get("E3_GPU_EXECUTION_ADMITTED") != "1":
            raise RuntimeError("PARENT_GPU_ADMISSION_REQUIRED")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible or visible in ("0", "7", "-1"):
            raise RuntimeError("PARENT_SINGLE_ADMITTED_GPU_REQUIRED_0_AND_7_FORBIDDEN")
        if not hasattr(signal, "setitimer"):
            raise RuntimeError("REAL_RUNTIME_REQUIRES_POSIX_HARD_TIMEOUT_SUPPORT")
        self.job.mkdir(parents=True, exist_ok=False)
        self.save_state()
        def timed_out(_sig, _frame):
            raise BudgetStop("FROZEN_BATCH_600S_OR_GLOBAL_CARD_DEADLINE")
        signal.signal(signal.SIGALRM, timed_out)
        self.arm_deadline()
        try:
            evidence, self.rows, self.pins = validate(self.contract, self.contract_path)
            put(self.job / "PRELOAD_VALIDATION.json", evidence)
            self.by_id = {r["page_id"]: r for r in self.rows}
            sys.path[:0] = [self.tele["archive_root"], self.tele["source_root"], *self.tele.get("extra_python_paths", [])]
            import TeleOCR.config as CONFIG
            self.config = CONFIG
            CONFIG.GPU_MEMORY_UTILIZATION = 0.5
            CONFIG.MAX_MODEL_LEN = 16384
            if CONFIG.LAYOUT_MODE != "Detection" or CONFIG.MAX_PIXELS != 8000 * 8000:
                raise RuntimeError("FROZEN_HELPER_CONFIG_CHANGED")
            os.environ["TELE_GPU_UTIL"] = "0.5"
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
            self.state.update(stage="LOADING", model_loading_started=True, GPU_started=True)
            self.save_state()
            import vllm
            from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
            from vllm.config import CompilationConfig
            from vllm.config.compilation import CompilationLevel, CUDAGraphMode
            client = TeleOCRMODEL_SERVICE.get_model("vllm-engine", self.tele["model"], None,
                enforce_eager=True, compilation_config=CompilationConfig(
                    level=CompilationLevel.NO_COMPILATION, cudagraph_mode=CUDAGraphMode.NONE))
            cfg = client.client.vllm_llm.llm_engine.vllm_config
            if Path(str(cfg.model_config.model)).resolve() != Path(self.tele["model"]).resolve():
                raise RuntimeError("ACTUAL_LOADED_MODEL_PATH_MISMATCH")
            if Path(str(cfg.model_config.tokenizer)).resolve() != Path(self.tele["model"]).resolve():
                raise RuntimeError("ACTUAL_LOADED_TOKENIZER_PATH_MISMATCH")
            effective = {"max_model_len": cfg.model_config.max_model_len,
                         "gpu_memory_utilization": cfg.cache_config.gpu_memory_utilization,
                         "enforce_eager": cfg.model_config.enforce_eager,
                         "compilation_level": int(cfg.compilation_config.level),
                         "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode).split(".")[-1]}
            for key, value in self.tele["config"].items():
                if key in effective and effective[key] != value:
                    raise RuntimeError("ACTUAL_LOADED_CONFIG_MISMATCH: " + key)
            if client.client.batch_size != 0 or client.batching_mode != "stepping" or client.incremental_priority:
                raise RuntimeError("FROZEN_HELPER_BATCHING_CHANGED")
            import TeleOCR.vlm_utils.TeleOCR_client as tc
            if plain(client.helper.prompts) != plain(tc.DEFAULT_PROMPTS) or plain(client.helper.sampling_params) != plain(tc.DEFAULT_SAMPLING_PARAMS):
                raise RuntimeError("FROZEN_SOURCE_PROMPT_OR_SAMPLING_CHANGED")
            if plain(client.helper.prompts) != self.tele["expected_prompts"] or plain(client.helper.sampling_params) != self.tele["expected_sampling"]:
                raise RuntimeError("FROZEN_CONTRACT_PROMPT_OR_SAMPLING_CHANGED")
            if client.client.system_prompt != self.tele["system_prompt"]:
                raise RuntimeError("FROZEN_CONTRACT_SYSTEM_PROMPT_CHANGED")
            helper = {k: plain(getattr(client.helper, k, UNKNOWN)) for k in (
                "prompts", "sampling_params", "layout_image_size", "min_image_edge", "max_image_edge_ratio",
                "simple_post_process", "handle_equation_block", "abandon_list", "abandon_paratext", "debug", "keep_four_numbers")}
            helper["keep_four_numbers_constructor_default"] = inspect.signature(type(client).__init__).parameters["keep_four_numbers"].default
            helper["keep_four_numbers_effective"] = "ABSENT_ATTRIBUTE_PARSE_CONVERT_DEFAULT_FALSE" if helper["keep_four_numbers"] == UNKNOWN else helper["keep_four_numbers"]
            if helper["keep_four_numbers_constructor_default"] is not True:
                raise RuntimeError("FROZEN_CLIENT_CONSTRUCTOR_KEEP_FOUR_NUMBERS_DEFAULT_CHANGED")
            for key, expected in self.tele.get("expected_helper", {}).items():
                if key == "keep_four_numbers" and helper[key] == UNKNOWN:
                    # Pinned helper forgets this constructor argument and parser uses _convert_bbox default False.
                    # Do not repair the historical behavior or invent an effective helper attribute.
                    default = inspect.signature(type(client).__init__).parameters[key].default
                    helper["keep_four_numbers_constructor_default"] = default
                    helper["keep_four_numbers_effective"] = "ABSENT_ATTRIBUTE_PARSE_CONVERT_DEFAULT_FALSE"
                    if default != expected:
                        raise RuntimeError("FROZEN_CONSTRUCTOR_DEFAULT_CHANGED: " + key)
                elif helper.get(key) != expected:
                    raise RuntimeError("FROZEN_CONTRACT_HELPER_PARAMETER_CHANGED: " + key)
            if helper["layout_image_size"] != [1036, 1036] or helper["min_image_edge"] != 28 or helper["max_image_edge_ratio"] != 50:
                raise RuntimeError("FROZEN_HELPER_IMAGE_PARAMETERS_CHANGED")
            sys.stdout.flush()
            sys.stderr.flush()
            log_path = getattr(sys.stdout, "name", None)
            log = Path(log_path).read_text(encoding="utf-8", errors="replace") if log_path and Path(log_path).is_file() else ""
            if "Using Flash Attention backend on V1 engine." not in log:
                raise RuntimeError("FROZEN_FLASH_ATTENTION_V1_NOT_OBSERVED_IN_ACTUAL_LOAD_LOG")
            if "Capturing CUDA graphs" in log or "torch.compile takes" in log:
                raise RuntimeError("FROZEN_EAGER_NO_COMPILE_GRAPH_LOAD_VIOLATED")
            put(self.job / "ACTUAL_CONFIG_READY.json", {"utc": utc(), "effective_config": effective,
                "actual_model_path": str(cfg.model_config.model), "actual_tokenizer_path": str(cfg.model_config.tokenizer),
                "runtime_versions": {"vllm": vllm.__version__, "torch": __import__("torch").__version__,
                                     "transformers": __import__("transformers").__version__},
                "helper": helper, "tokenizer_class": str(type(client.client.tokenizer)),
                "tokenizer_chat_template_sha256": hashlib.sha256(str(client.client.tokenizer.chat_template).encode()).hexdigest(),
                "processor_files_preload_hash_verified": True, "image_tokens_are_text_tokens": False,
                "runtime_attention_backend": "FLASH_ATTENTION_V1_LOG_VERIFIED",
                "runtime_compile_graph_absence_log_verified": True, "GPU_identity": visible})
            self.check_loaded_modules()
            self.install_observers(client, vllm.LLM)
            for target in ("BASE", "B0"):
                for sub in ("markdown", "middle"):
                    (self.job / target / sub).mkdir(parents=True)
            for index, pos in enumerate(range(0, 20, 16)):
                rows = self.rows[pos:pos + 16]
                directory = self.job / "BATCHES" / f"{index:04d}"
                directory.mkdir(parents=True)
                bound = [{**row, "id": row["page_id"]} for row in rows]
                put(directory / "INPUTS.json", {"bound_rows": bound, "fresh_original_input_only": True})
                self.state["batch"] = index
                self.page_ids = [r["page_id"] for r in rows]
                self.invoke_stage(directory, rows, "NATIVE", "run_tele_img.py", ["--out", str(directory / "NATIVE"), "--chunk", "16"])
                for row in rows:
                    for sub, ext in (("markdown", "md"), ("middle", "json")):
                        path = directory / "NATIVE" / sub / (row["page_id"] + "." + ext)
                        shutil.copyfile(path, self.job / "BASE" / sub / path.name)
                self.invoke_stage(directory, rows, "FORMULA125", "stack_reread_v1.py", ["--run", str(directory / "NATIVE"), "--out", str(directory / "FORMULA125"), "--kinds", "equation", "--scale", "1.25", "--chunk", "32"])
                self.invoke_stage(directory, rows, "GUARD", "stack_guard_v1.py", ["--run", str(directory / "FORMULA125"), "--fallback", str(directory / "NATIVE"), "--out", str(directory / "GUARD"), "--scales", "1.0,0.75,1.5"])
                view = directory / "GUARD_VIEW"
                for sub in ("markdown", "middle"):
                    (view / sub).mkdir(parents=True)
                for row in rows:
                    for sub, ext in (("markdown", "md"), ("middle", "json")):
                        path = next(directory / s / sub / (row["page_id"] + "." + ext) for s in ("GUARD", "FORMULA125", "NATIVE")
                                    if (directory / s / sub / (row["page_id"] + "." + ext)).exists())
                        shutil.copyfile(path, view / sub / path.name)
                self.invoke_stage(directory, rows, "TABLE1X", "stack_reread_v1.py", ["--run", str(view), "--out", str(directory / "TABLE1X"), "--kinds", "table", "--scale", "1.0", "--chunk", "32"])
                for row in rows:
                    record = {"utc": utc(), "page_id": row["page_id"], "image_sha256": row["image_sha256"], "batch": index}
                    for sub, ext in (("markdown", "md"), ("middle", "json")):
                        path = next(directory / s / sub / (row["page_id"] + "." + ext) for s in ("TABLE1X", "GUARD_VIEW")
                                    if (directory / s / sub / (row["page_id"] + "." + ext)).exists())
                        target = self.job / "B0" / sub / path.name
                        shutil.copyfile(path, target)
                        record["b0_" + sub + "_sha256"] = sha(target)
                        record["base_" + sub + "_sha256"] = sha(self.job / "BASE" / sub / path.name)
                    append(self.job / "PAGE_B0_CHECKPOINTS.jsonl", record)
                    md_path = self.job / "B0" / "markdown" / (row["page_id"] + ".md")
                    terminal = "COMPLETE" if md_path.read_text(encoding="utf-8").strip() else "RETURNED_EMPTY"
                    append(self.job / "PAGE_PAIRED_CHECKPOINTS.jsonl", {
                        "utc": utc(), "page_id": row["page_id"], "input_image_sha256": row["image_sha256"],
                        "b0_md_sha256": record["b0_markdown_sha256"],
                        "b0_middle_sha256": record["b0_middle_sha256"],
                        "base_md_sha256": record["base_markdown_sha256"],
                        "base_middle_sha256": record["base_middle_sha256"],
                        "stages_completed": list(STAGES), "status": terminal,
                        "run_contract_sha256": self.run_contract_sha256,
                        "stage_accepted_adoption_map": UNKNOWN,
                        "adoption_uncertainty": "EXACT_CURRENT_TARGET_RAW_TO_FINAL_MIDDLE_ASSOCIATION_REQUIRED_NO_INVENTED_SLOT"})
                    self.state["completed_pages"].append(row["page_id"])
                self.save_state()
            self.check_loaded_modules()
            for p in self.pins:
                if Path(self.tele["archive_root"]).resolve() in Path(p).parents or Path(self.tele["source_root"]).resolve() in Path(p).parents:
                    require_pin(p, self.pins)
            self.state["stage"] = "COMPLETE"
            self.save_state()
            put(self.job / "WORKER_EXIT.json", {"utc": utc(), "schema": SCHEMA, "exit": 0,
                "pages": len(self.state["completed_pages"]), "submitted_requests": self.budget.submitted,
                "completed_requests": self.state["completed_requests"], "wall_seconds": time.monotonic() - self.started,
                "no_scoring_or_reference_access": True, "fresh_four_stage_b0": True})
        except BaseException as error:
            self.state.update(error_type=type(error).__name__, error=str(error), terminal_status="ERROR_OR_BUDGET_STOP")
            self.save_state()
            put(self.job / "WORKER_EXIT.json", {"utc": utc(), "schema": SCHEMA, "exit": 1,
                "stage": self.state["stage"], "completed_pages": self.state["completed_pages"],
                "submitted_requests": self.budget.submitted, "completed_requests": self.state["completed_requests"],
                "error_type": type(error).__name__, "error": str(error), "wall_seconds": time.monotonic() - self.started,
                "fixed20_denominator_retained_by_supervisor": True, "no_implicit_retry": True})
            traceback.print_exc()
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def mock(contract, contract_path, job, deadline, maximum):
    """CPU-only budget/identity harness, explicitly synthetic, not original-stage model validation."""
    evidence, rows, _ = validate(contract, contract_path, verify_files=False)
    job = Path(job)
    job.mkdir(parents=True, exist_ok=False)
    budget = RequestBudget(deadline, maximum, 600)
    records = []
    for index, pos in enumerate(range(0, 20, 16)):
        batch = rows[pos:pos + 16]
        for stage in STAGES:
            budget.reserve(len(batch))
            append(job / "STAGES.jsonl", {"batch": index, "stage": stage, "synthetic": True,
                   "status": "COMPLETE", "GPU_started": False})
        records.extend({"page_id": r["page_id"], "input_image_sha256": r["image_sha256"],
                        "status": "SYNTHETIC_CONTROL_COMPLETE", "GPU_started": False} for r in batch)
    evidence.update(status="CPU_SYNTHETIC_CONTROL_PASS", model_stage_execution=False,
                    terminal_pages=records, synthetic_request_budget_reserved=budget.submitted,
                    root_full_orchestration_acceptance=False)
    put(job / "STATE.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    contract = read_json(args.contract)
    if args.validate_only:
        evidence, _, _ = validate(contract, args.contract)
        put(Path(args.job) / "CPU_VALIDATION.json", evidence)
        print(json.dumps({k: evidence[k] for k in ("status", "pages", "documents", "verified_images", "verified_external_pins", "GPU_started")}, ensure_ascii=True))
        return
    if args.mock:
        evidence = mock(contract, args.contract, args.job, args.deadline_epoch, args.max_requests)
        print(json.dumps({"status": evidence["status"], "pages": len(evidence["terminal_pages"]), "GPU_started": False}))
        return
    # Preserve vllm/source warnings containing answers in owned private log only.
    job = Path(args.job)
    if job.exists():
        raise RuntimeError("REFUSE_IMPLICIT_RESTART_OR_NONFRESH_OUTPUT")
    job.parent.mkdir(parents=True, exist_ok=True)
    private_log = job.parent / (job.name + ".PRIVATE_RUNTIME.log")
    with private_log.open("x", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            Worker(contract, args.contract, job, args.deadline_epoch, args.max_requests).execute()


if __name__ == "__main__":
    main()
