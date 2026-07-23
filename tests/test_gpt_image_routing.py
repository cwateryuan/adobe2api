import unittest

from fastapi import HTTPException

from core.models import (
    DEFAULT_MODEL_ID,
    GPT_IMAGE_MODEL_ID,
    GPT_IMAGE_PIXEL_SIZES,
    MODEL_CATALOG,
    build_image_payload_candidates,
    resolve_ratio_and_resolution,
)


class GptImageRoutingTests(unittest.TestCase):
    def resolve(self, size=None, **data):
        if size is not None:
            data["size"] = size
        return resolve_ratio_and_resolution(data, GPT_IMAGE_MODEL_ID)

    def assert_invalid_size(self, size):
        with self.assertRaises(HTTPException) as raised:
            self.resolve(size)
        self.assertEqual(raised.exception.status_code, 400)

    def test_catalog_shows_alias_and_hides_legacy_gpt_models(self):
        visible_models = {
            model_id
            for model_id, conf in MODEL_CATALOG.items()
            if not bool(conf.get("hidden", False))
        }
        legacy_models = {
            model_id
            for model_id in MODEL_CATALOG
            if model_id.startswith("firefly-gpt-image-")
        }

        self.assertIn(GPT_IMAGE_MODEL_ID, visible_models)
        self.assertTrue(legacy_models)
        self.assertTrue(legacy_models.isdisjoint(visible_models))
        self.assertIn(DEFAULT_MODEL_ID, visible_models)

    def test_near_square_routes_to_one_k_square(self):
        for size in ("1026x1026", "1026X1026"):
            with self.subTest(size=size):
                self.assertEqual(self.resolve(size), ("1:1", "1K", GPT_IMAGE_MODEL_ID))

    def test_every_exact_supported_size_keeps_its_tier_and_ratio(self):
        for level, ratios in GPT_IMAGE_PIXEL_SIZES.items():
            for ratio, dimensions in ratios.items():
                size = f'{dimensions["width"]}x{dimensions["height"]}'
                with self.subTest(level=level, ratio=ratio, size=size):
                    self.assertEqual(
                        self.resolve(size),
                        (ratio, level, GPT_IMAGE_MODEL_ID),
                    )

    def test_pixel_count_tie_uses_lower_price_tier(self):
        # This area is halfway between the largest 1K and smallest 2K canvas.
        _ratio, level, _model_id = self.resolve("1024x2312")
        self.assertEqual(level, "1K")

    def test_tier_is_selected_before_aspect_ratio(self):
        ratio, level, _model_id = self.resolve("2500x1600")
        self.assertEqual(level, "2K")
        self.assertEqual(ratio, "3:2")

    def test_maximum_edge_is_accepted(self):
        self.assertEqual(
            self.resolve("3840x3840"),
            ("1:1", "4K", GPT_IMAGE_MODEL_ID),
        )

    def test_oversized_edges_are_rejected(self):
        for size in ("3841x1024", "1024x3841"):
            with self.subTest(size=size):
                self.assert_invalid_size(size)

    def test_invalid_formats_are_rejected(self):
        invalid_sizes = (
            "1026 X 1026",
            " 1026x1026",
            "1026x1026 ",
            "1024.0x1024",
            "+1024x1024",
            "-1x1024",
            "0x1024",
            "1024x0",
            "1024xx1024",
            "1024*1024",
            "AUTO",
            "",
            f'{"9" * 5000}x1',
            1024,
            {"width": 1024, "height": 1024},
        )
        for size in invalid_sizes:
            with self.subTest(size=size):
                self.assert_invalid_size(size)

    def test_missing_size_and_auto_default_to_one_k_square(self):
        expected = ("1:1", "1K", GPT_IMAGE_MODEL_ID)
        self.assertEqual(self.resolve(), expected)
        self.assertEqual(self.resolve("auto"), expected)

    def test_size_overrides_aspect_ratio_and_quality(self):
        self.assertEqual(
            self.resolve("1026x1026", aspect_ratio="21:9", quality="4k"),
            ("1:1", "1K", GPT_IMAGE_MODEL_ID),
        )

    def test_legacy_gpt_model_keeps_fixed_route_and_ignores_size(self):
        legacy_model = "firefly-gpt-image-4k-4x5"
        self.assertEqual(
            resolve_ratio_and_resolution({"size": "not-a-size"}, legacy_model),
            ("4:5", "4K", legacy_model),
        )

    def test_non_gpt_model_keeps_existing_fixed_route(self):
        model_id = "firefly-nano-banana-pro-2k-16x9"
        self.assertEqual(
            resolve_ratio_and_resolution(
                {"size": "1024x1024", "aspect_ratio": "1:1", "quality": "4k"},
                model_id,
            ),
            ("16:9", "2K", model_id),
        )

    def test_payload_uses_the_routed_tier_and_size(self):
        ratio, level, _model_id = self.resolve("1026X1026")
        payload = build_image_payload_candidates(
            prompt="test",
            aspect_ratio=ratio,
            output_resolution=level,
            upstream_model_id="gpt-image",
            upstream_model_version="2",
        )[0]

        self.assertEqual(payload["outputResolution"], "1K")
        self.assertEqual(payload["size"], {"width": 1024, "height": 1024})
        self.assertEqual(payload["modelSpecificPayload"]["size"], "1024x1024")


if __name__ == "__main__":
    unittest.main()
