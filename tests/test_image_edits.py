import base64
import io
import logging
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.routes.generation import build_generation_router
from core.models import (
    MODEL_CATALOG,
    SUPPORTED_RATIOS,
    VIDEO_MODEL_CATALOG,
    resolve_model,
    resolve_ratio_and_resolution,
)


class TestQuotaError(Exception):
    pass


class TestAuthError(Exception):
    pass


class TestUpstreamError(Exception):
    pass


def make_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(output, format="PNG")
    return output.getvalue()


class FakeClient:
    generate_timeout = 30
    gpt_image_quality = "high"
    token_rotation_strategy = "round_robin"

    def __init__(self, image_bytes: bytes):
        self.image_bytes = image_bytes
        self.uploads = []
        self.generations = []

    def upload_image(self, token, image_bytes, image_mime):
        self.uploads.append((token, image_bytes, image_mime))
        return f"image-{len(self.uploads)}"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        return self.image_bytes, {"progress": 100}


class ImageEditsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.generated_dir = Path(self.temp_dir.name)
        self.png_bytes = make_png()
        self.client_impl = FakeClient(self.png_bytes)
        self.written_files = []
        router = build_generation_router(
            store=None,
            token_manager=None,
            client=self.client_impl,
            generated_dir=self.generated_dir,
            model_catalog=MODEL_CATALOG,
            video_model_catalog=VIDEO_MODEL_CATALOG,
            supported_ratios=SUPPORTED_RATIOS,
            resolve_model=resolve_model,
            resolve_ratio_and_resolution=resolve_ratio_and_resolution,
            require_service_api_key=lambda request: None,
            set_request_task_progress=lambda *args, **kwargs: None,
            run_with_token_retries=lambda request, operation_name, run_once, **kwargs: run_once(
                "test-token"
            ),
            set_request_error_detail=lambda *args, **kwargs: "test-error",
            set_request_preview=lambda *args, **kwargs: None,
            public_image_url=lambda request, job_id: f"http://test/{job_id}.png",
            public_generated_url=lambda request, filename: f"http://test/{filename}",
            resolve_video_options=lambda data: (True, "", "frame"),
            load_input_images=lambda messages: [],
            prepare_video_source_image=lambda data, ratio, resolution: (
                data,
                "image/png",
            ),
            video_ext_from_meta=lambda meta: "mp4",
            extract_prompt_from_messages=lambda messages: "",
            sse_chat_stream=lambda payload: iter(()),
            on_generated_file_written=lambda path, old, new: self.written_files.append(
                (path, old, new)
            ),
            quota_error_cls=TestQuotaError,
            auth_error_cls=TestAuthError,
            upstream_temp_error_cls=TestUpstreamError,
            logger=logging.getLogger("test.image_edits"),
        )
        app = FastAPI()
        app.include_router(router)
        self.http = TestClient(app)

    def tearDown(self):
        self.http.close()
        self.temp_dir.cleanup()

    def test_multipart_request_supports_n_and_webp(self):
        response = self.http.post(
            "/v1/images/edits",
            data={
                "model": "gpt-image-2",
                "prompt": "change the subject",
                "aspect_ratio": "21:9",
                "size": "1254x1254",
                "quality": "high",
                "output_format": "webp",
                "n": "2",
                "response_format": "url",
            },
            files={"image": ("input.png", self.png_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["model"], "gpt-image-2")
        self.assertEqual(len(payload["data"]), 2)
        self.assertTrue(all(item["url"].endswith(".webp") for item in payload["data"]))
        self.assertEqual(len(self.client_impl.uploads), 1)
        self.assertEqual(len(self.client_impl.generations), 2)
        self.assertEqual(self.client_impl.generations[0]["aspect_ratio"], "1:1")
        self.assertEqual(self.client_impl.generations[0]["output_resolution"], "1K")
        self.assertEqual(self.client_impl.generations[0]["quality_level"], "high")
        self.assertEqual(self.client_impl.generations[0]["source_image_ids"], ["image-1"])
        self.assertEqual(len(self.written_files), 2)
        for path, _old_size, _new_size in self.written_files:
            with Image.open(path) as image:
                self.assertEqual(image.format, "WEBP")

    def test_multipart_request_uploads_image_and_uses_size_route(self):
        response = self.http.post(
            "/v1/images/edits",
            data={
                "model": "gpt-image-2",
                "prompt": "outpaint the room",
                "n": "1",
                "quality": "high",
                "size": "1344x1792",
            },
            files={"image": ("input.png", self.png_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["data"]), 1)
        self.assertTrue(payload["data"][0]["url"].endswith(".png"))
        self.assertEqual(len(self.client_impl.uploads), 1)
        self.assertEqual(self.client_impl.uploads[0][2], "image/png")
        self.assertEqual(self.client_impl.generations[0]["aspect_ratio"], "3:4")
        self.assertEqual(self.client_impl.generations[0]["output_resolution"], "2K")

    def test_b64_json_response(self):
        response = self.http.post(
            "/v1/images/edits",
            data={
                "model": "gpt-image-2",
                "prompt": "edit",
                "size": "1024X1024",
                "response_format": "b64_json",
            },
            files={"image": ("input.png", self.png_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        encoded = response.json()["data"][0]["b64_json"]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.format, "PNG")

    def test_invalid_size_returns_openai_error(self):
        response = self.http.post(
            "/v1/images/edits",
            data={
                "model": "gpt-image-2",
                "prompt": "edit",
                "size": "1024 X 1024",
            },
            files={"image": ("input.png", self.png_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(len(self.client_impl.generations), 0)

    def test_multipart_requires_an_image(self):
        response = self.http.post(
            "/v1/images/edits",
            data={"model": "gpt-image-2", "prompt": "edit"},
            files={"other": ("ignored.txt", b"ignored", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    def test_json_url_input_is_not_supported(self):
        response = self.http.post(
            "/v1/images/edits",
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "size": "1024x1024",
                "images": [{"image_url": "https://example.com/input.png"}],
            },
        )

        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
