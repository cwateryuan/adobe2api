from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from PIL import Image
from starlette.datastructures import UploadFile


MAX_EDIT_IMAGES = 6
MAX_EDIT_IMAGE_BYTES = 10 * 1024 * 1024
OUTPUT_FORMAT_EXTENSIONS = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "webp": "webp",
}


def _image_mime(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = str(image.format or "").upper()
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid multipart image") from exc

    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    mime_type = mime_types.get(image_format)
    if mime_type is None:
        raise HTTPException(
            status_code=400,
            detail="multipart image must be JPEG, PNG, or WebP",
        )
    return mime_type


async def parse_image_edit_request(
    request: Request,
) -> tuple[dict, list[tuple[bytes, str]]]:
    content_type = str(request.headers.get("content-type") or "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be multipart/form-data",
        )

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid multipart body") from exc

    data: dict[str, Any] = {}
    uploads: list[UploadFile] = []
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key in {"image", "image[]", "images"}:
                uploads.append(value)
            else:
                await value.close()
            continue
        data[key] = value

    if not uploads:
        raise HTTPException(status_code=400, detail="at least one image is required")
    if len(uploads) > MAX_EDIT_IMAGES:
        for upload in uploads:
            await upload.close()
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_EDIT_IMAGES} images are supported",
        )

    images: list[tuple[bytes, str]] = []
    try:
        for upload in uploads:
            image_bytes = await upload.read(MAX_EDIT_IMAGE_BYTES + 1)
            if not image_bytes:
                raise HTTPException(status_code=400, detail="multipart image is empty")
            if len(image_bytes) > MAX_EDIT_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="image too large, max 10MB")
            images.append((image_bytes, _image_mime(image_bytes)))
    finally:
        for upload in uploads:
            await upload.close()
    return data, images


def parse_image_edit_options(data: dict) -> tuple[str, int, str, str, str]:
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    raw_n = data.get("n", 1)
    if isinstance(raw_n, bool):
        raise HTTPException(status_code=400, detail="n must be an integer between 1 and 10")
    try:
        n = int(raw_n)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="n must be an integer between 1 and 10"
        ) from exc
    if n < 1 or n > 10:
        raise HTTPException(status_code=400, detail="n must be an integer between 1 and 10")

    output_format = str(data.get("output_format") or "png").lower()
    if output_format not in OUTPUT_FORMAT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="output_format must be png, jpeg, jpg, or webp",
        )

    response_format = str(data.get("response_format") or "url").lower()
    if response_format not in {"url", "b64_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be url or b64_json",
        )

    quality = str(data.get("quality") or "auto").lower()
    if quality not in {"auto", "low", "medium", "high"}:
        raise HTTPException(
            status_code=400,
            detail="quality must be auto, low, medium, or high",
        )
    return prompt, n, output_format, response_format, quality


def convert_generated_image(
    source_path: Path,
    destination_path: Path,
    output_format: str,
) -> None:
    try:
        with Image.open(source_path) as image:
            image.load()
            if output_format in {"jpeg", "jpg"}:
                if image.mode in {"RGBA", "LA"}:
                    rgba = image.convert("RGBA")
                    converted = Image.new("RGB", rgba.size, "white")
                    converted.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    converted = image.convert("RGB")
                converted.save(destination_path, format="JPEG", quality=95)
            elif output_format == "webp":
                image.save(destination_path, format="WEBP", quality=95, method=6)
            else:
                image.save(destination_path, format="PNG", optimize=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="failed to encode generated image") from exc
