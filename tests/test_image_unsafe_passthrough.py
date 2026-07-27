import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from core.adobe_client import AdobeClient, AdobeRequestError, UpstreamTemporaryError


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("response is not JSON")
        return self._payload


class ImageUnsafePassthroughTests(unittest.TestCase):
    unsafe_payload = {
        "error_code": "image_unsafe",
        "message": (
            "The generated images appear to be unsafe. "
            "Try modifying the prompts or the seeds."
        ),
    }

    def make_client(self, submit_response, poll_response=None, candidates=None):
        client = AdobeClient()
        client._build_payload_candidates = Mock(return_value=candidates or [{}])
        client._post_json = Mock(return_value=submit_response)
        client._get = Mock(return_value=poll_response)
        return client

    def assert_unsafe_error(self, raised):
        error = raised.exception
        self.assertIsInstance(error, AdobeRequestError)
        self.assertNotIsInstance(error, UpstreamTemporaryError)
        self.assertEqual(error.status_code, 451)
        self.assertEqual(error.error_type, "invalid_request_error")
        self.assertEqual(error.user_message, self.unsafe_payload["message"])

    def test_poll_image_unsafe_is_a_non_retryable_451(self):
        submit_response = FakeResponse(
            200,
            {"links": {"result": "https://example.test/jobs/123"}},
        )
        poll_response = FakeResponse(451, self.unsafe_payload)
        client = self.make_client(submit_response, poll_response)

        with self.assertRaises(AdobeRequestError) as raised:
            client.generate(token="token", prompt="prompt")

        self.assert_unsafe_error(raised)
        client._post_json.assert_called_once()
        client._get.assert_called_once()

    def test_submit_image_unsafe_does_not_try_payload_fallbacks(self):
        submit_response = FakeResponse(451, self.unsafe_payload)
        client = self.make_client(
            submit_response,
            candidates=[{"candidate": 1}, {"candidate": 2}],
        )

        with self.assertRaises(AdobeRequestError) as raised:
            client.generate(token="token", prompt="prompt")

        self.assert_unsafe_error(raised)
        client._post_json.assert_called_once()
        client._get.assert_not_called()

    def test_submit_transport_does_not_replay_image_unsafe_response(self):
        unsafe_response = FakeResponse(451, self.unsafe_payload)
        session = MagicMock()
        session.post.return_value = unsafe_response
        client = AdobeClient()
        client._session = Mock(return_value=session)

        with patch("core.adobe_client.requests.post") as requests_post:
            response = client._post_json(
                "https://example.test/generate",
                headers={"authorization": "Bearer token"},
                payload={"prompt": "prompt"},
            )

        self.assertIs(response, unsafe_response)
        session.post.assert_called_once()
        requests_post.assert_not_called()

    def test_other_451_responses_remain_temporary_errors(self):
        submit_response = FakeResponse(
            200,
            {"links": {"result": "https://example.test/jobs/123"}},
        )
        poll_response = FakeResponse(
            451,
            {"error_code": "upstream_unavailable", "message": "try later"},
        )
        client = self.make_client(submit_response, poll_response)

        with self.assertRaises(UpstreamTemporaryError) as raised:
            client.generate(token="token", prompt="prompt")

        self.assertEqual(raised.exception.status_code, 451)


if __name__ == "__main__":
    unittest.main()
