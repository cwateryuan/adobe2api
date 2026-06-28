import unittest

from api.routes.admin import build_admin_router
from api.schemas import ConfigUpdateRequest


class DummyConfigManager:
    def __init__(self):
        self.config = {
            "api_key": "test",
            "admin_session_secret": "hidden",
            "generated_max_size_mb": 1024,
            "generated_prune_size_mb": 200,
            "flaresolverr_enabled": True,
            "flaresolverr_url": "http://127.0.0.1:8191/v1",
            "flaresolverr_max_timeout_ms": 60000,
            "flaresolverr_use_proxy": True,
            "flaresolverr_session": "",
            "flaresolverr_trigger_status_codes": [408, 429, 451, 503],
        }

    def get_all(self):
        return dict(self.config)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def update_all(self, data):
        self.config.update(data)


class AdminConfigTests(unittest.TestCase):
    def _handlers(self, manager):
        router = build_admin_router(
            static_dir=None,
            token_manager=None,
            config_manager=manager,
            refresh_manager=None,
            log_store=None,
            error_store=None,
            live_log_store=None,
            require_admin_auth=lambda request: None,
            is_admin_authenticated=lambda request: True,
            apply_client_config=lambda: None,
            get_generated_storage_stats=lambda: {},
        )
        return {
            route.path + ":" + ",".join(sorted(route.methods)): route.endpoint
            for route in router.routes
        }

    def test_get_config_exposes_flaresolverr_fields(self):
        manager = DummyConfigManager()
        handlers = self._handlers(manager)

        payload = handlers["/api/v1/config:GET"](request=None)

        self.assertTrue(payload["flaresolverr_enabled"])
        self.assertEqual(payload["flaresolverr_url"], "http://127.0.0.1:8191/v1")
        self.assertEqual(payload["flaresolverr_trigger_status_codes"], [408, 429, 451, 503])
        self.assertNotIn("admin_session_secret", payload)

    def test_update_config_saves_flaresolverr_fields(self):
        manager = DummyConfigManager()
        handlers = self._handlers(manager)

        payload = handlers["/api/v1/config:PUT"](
            ConfigUpdateRequest(
                flaresolverr_enabled=True,
                flaresolverr_url="http://host.docker.internal:8191/v1",
                flaresolverr_max_timeout_ms=120000,
                flaresolverr_use_proxy=False,
                flaresolverr_session="adobe",
                flaresolverr_trigger_status_codes=[408, 503],
            ),
            request=None,
        )

        self.assertTrue(payload["flaresolverr_enabled"])
        self.assertEqual(
            payload["flaresolverr_url"], "http://host.docker.internal:8191/v1"
        )
        self.assertEqual(payload["flaresolverr_max_timeout_ms"], 120000)
        self.assertFalse(payload["flaresolverr_use_proxy"])
        self.assertEqual(payload["flaresolverr_session"], "adobe")
        self.assertEqual(payload["flaresolverr_trigger_status_codes"], [408, 503])


if __name__ == "__main__":
    unittest.main()
