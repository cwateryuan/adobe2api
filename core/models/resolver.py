from __future__ import annotations

import math
import re
from typing import Optional

from fastapi import HTTPException

from .catalog import (
    DEFAULT_MODEL_ID,
    GPT_IMAGE_MODEL_ID,
    GPT_IMAGE_PIXEL_SIZES,
    MODEL_CATALOG,
    SUPPORTED_RATIOS,
)


_GPT_IMAGE_SIZE_RE = re.compile(r"([0-9]+)[xX]([0-9]+)\Z")
_GPT_IMAGE_MAX_EDGE = 3840
_GPT_IMAGE_LEVELS = ("1K", "2K", "4K")


def resolve_model(model_id: Optional[str]) -> dict:
    if not model_id:
        return MODEL_CATALOG[DEFAULT_MODEL_ID]
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_id}")
    return MODEL_CATALOG[model_id]


def ratio_from_size(size: str) -> str:
    mapping = {
        "1024x1024": "1:1",
        "1536x1536": "1:1",
        "2048x2048": "1:1",
        "1024x1792": "9:16",
        "1536x2752": "9:16",
        "1792x1024": "16:9",
        "2752x1536": "16:9",
        "2048x1536": "4:3",
        "1536x2048": "3:4",
    }
    return mapping.get(str(size or "").strip(), "1:1")


def resolve_gpt_image_size(size: object) -> tuple[str, str]:
    if size is None or size == "auto":
        size = "1024x1024"
    if not isinstance(size, str):
        raise HTTPException(status_code=400, detail="size must use WIDTHxHEIGHT format")

    match = _GPT_IMAGE_SIZE_RE.fullmatch(size)
    if match is None:
        raise HTTPException(status_code=400, detail="size must use WIDTHxHEIGHT format")

    try:
        width, height = (int(value) for value in match.groups())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="size dimensions must be valid integers"
        ) from exc
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size dimensions must be positive")
    if width > _GPT_IMAGE_MAX_EDGE or height > _GPT_IMAGE_MAX_EDGE:
        raise HTTPException(
            status_code=400,
            detail=f"size dimensions must not exceed {_GPT_IMAGE_MAX_EDGE}",
        )

    requested_pixels = width * height
    output_resolution = min(
        _GPT_IMAGE_LEVELS,
        key=lambda level: min(
            abs(requested_pixels - dimensions["width"] * dimensions["height"])
            for dimensions in GPT_IMAGE_PIXEL_SIZES[level].values()
        ),
    )

    requested_ratio = width / height
    ratio = min(
        GPT_IMAGE_PIXEL_SIZES[output_resolution],
        key=lambda candidate: abs(
            math.log(
                requested_ratio
                / (
                    GPT_IMAGE_PIXEL_SIZES[output_resolution][candidate]["width"]
                    / GPT_IMAGE_PIXEL_SIZES[output_resolution][candidate]["height"]
                )
            )
        ),
    )
    return ratio, output_resolution


def resolve_ratio_and_resolution(
    data: dict, model_id: Optional[str]
) -> tuple[str, str, str]:
    if model_id == GPT_IMAGE_MODEL_ID:
        ratio, output_resolution = resolve_gpt_image_size(data.get("size"))
        return ratio, output_resolution, GPT_IMAGE_MODEL_ID

    ratio = str(data.get("aspect_ratio") or "").strip() or ratio_from_size(
        data.get("size", "1024x1024")
    )
    if ratio not in SUPPORTED_RATIOS:
        ratio = "1:1"

    resolved_model_id = model_id or DEFAULT_MODEL_ID
    if resolved_model_id not in MODEL_CATALOG:
        resolved_model_id = DEFAULT_MODEL_ID
    model_conf = MODEL_CATALOG[resolved_model_id]

    output_resolution = model_conf["output_resolution"]
    if not model_id:
        quality = str(data.get("quality", "2k")).lower()
        if quality in ("4k", "ultra"):
            output_resolution = "4K"
        elif quality in ("hd", "2k"):
            output_resolution = "2K"
        else:
            output_resolution = "1K"

    model_ratio = model_conf.get("aspect_ratio")
    if model_ratio:
        ratio = model_ratio

    return ratio, output_resolution, resolved_model_id
