"""Paddle serialization boundary; no model imports, cached predictions or GT.

use_layout_detection=False is not rejected: supplied geometry is validated.
This adapter proves structure only, not model identity or inference readiness.
"""
import json
import math


def materialize_one(outputs):
    """Consume predict() generators instead of json.dumps(..., default=str)."""
    results = list(outputs)
    if len(results) != 1:
        raise ValueError('ONE_IMAGE_REQUIRES_ONE_RESULT')
    result = results[0]
    data = getattr(result, 'json', result)
    if callable(data):
        data = data()
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError('PADDLE_RESULT_NOT_JSON_OBJECT')
    data = data.get('res', data)
    markdown = getattr(result, 'markdown', None)
    if callable(markdown):
        markdown = markdown()
    if isinstance(markdown, dict):
        markdown = markdown.get('markdown_texts')
    if markdown is not None and not isinstance(markdown, str):
        raise ValueError('PADDLE_MARKDOWN_NOT_STRING')
    return data, markdown


def validate_donor(data, expected_size):
    """Validate exactly the original R3 consumer fields, without inventing boxes."""
    if not isinstance(data, dict):
        raise ValueError('DONOR_NOT_OBJECT')
    size = [data.get('width'), data.get('height')]
    if size != list(expected_size) or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in size):
        raise ValueError('DONOR_IMAGE_DIMENSIONS_MISMATCH')
    blocks = data.get('parsing_res_list')
    if not isinstance(blocks, list):
        raise ValueError('DONOR_PARSING_RES_LIST_MISSING')
    for b in blocks:
        if not isinstance(b, dict) or not isinstance(b.get('block_label'), str) or not isinstance(b.get('block_content'), str):
            raise ValueError('DONOR_BLOCK_LABEL_OR_CONTENT_MISSING')
        box = b.get('block_bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError('DONOR_BOX_INVALID')
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise ValueError('DONOR_BOX_OUTSIDE_IMAGE_OR_NO_AREA')
    return {'schema_valid': True, 'blocks': len(blocks), 'size': size,
            'unique_overlap_and_exact_patch_still_required': True,
            'model_identity': 'UNKNOWN', 'GPU_smoke': 'NOT_RUN'}
