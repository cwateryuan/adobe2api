import unittest

import requests

from core.adobe_client import AdobeClient


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class AdobeClientTemporaryErrorTests(unittest.TestCase):
    def test_408_is_temporary_by_default(self):
        client = AdobeClient()

        self.assertIn(408, client.retry_on_status_codes)
        self.assertTrue(
            client._is_temporary_response(
                FakeResponse(408, '{"error_code":"timeout_error"}')
            )
        )

    def test_timeout_error_body_is_temporary_even_without_retry_status(self):
        client = AdobeClient()

        self.assertTrue(
            client._is_temporary_response(
                FakeResponse(
                    400,
                    '{"error_code":"timeout_error","message":"system under load"}',
                )
            )
        )

    def test_408_post_uses_flaresolverr_fallback_by_default(self):
        client = AdobeClient()
        calls = []
        original_post = requests.post

        def fake_post(url, **kwargs):
            calls.append({"url": url, "kwargs": kwargs})
            resp = requests.Response()
            if url == "http://127.0.0.1:8191/v1":
                resp.status_code = 200
                resp._content = (
                    b'{"status":"ok","solution":{"userAgent":"solver-ua",'
                    b'"cookies":[{"name":"cf_clearance","value":"ok"}]}}'
                )
                return resp

            if len(calls) == 1:
                resp.status_code = 408
                resp._content = (
                    b'{"error_code":"timeout_error","message":"system under load"}'
                )
                return resp

            resp.status_code = 200
            resp._content = b'{"links":{"result":{"href":"https://example.test/job"}}}'
            return resp

        try:
            requests.post = fake_post
            resp = client._post_json(
                "https://firefly-3p.ff.adobe.io/v2/3p-images/generate-async",
                headers={"user-agent": "original", "content-type": "application/json"},
                payload={"prompt": "test"},
            )
        finally:
            requests.post = original_post

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1]["url"], "http://127.0.0.1:8191/v1")
        replay_headers = calls[2]["kwargs"]["headers"]
        self.assertEqual(replay_headers["user-agent"], "solver-ua")
        self.assertEqual(replay_headers["cookie"], "cf_clearance=ok")


if __name__ == "__main__":
    unittest.main()
