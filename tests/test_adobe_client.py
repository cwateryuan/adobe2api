import unittest

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


if __name__ == "__main__":
    unittest.main()
