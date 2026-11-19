import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

import numpy as np
import torch
from torch import nn

from dipa.attack import optimize, cosine_scores
from dipa.io import load_image, save_image
from dipa.models import FaceEncoder, map_state_dict, load_models
from dipa.patch import apply_patch, median_pool, render_patch, total_variation
from prepare_face import similarity_transform, TEMPLATE_112
from summarize_camera import summarize


torch.set_num_threads(1)


class ColorBackbone(nn.Module):
    def forward(self, image):
        average = image.mean(dim=(2, 3))
        return torch.cat([average, torch.ones_like(average[:, :1]) * 0.2], dim=1)


def toy_models():
    return [FaceEncoder(ColorBackbone(), name, mean=(0, 0, 0), std=(1, 1, 1))
            for name in ("arcface", "cosface", "magface")]


def config():
    return {"steps": 12, "learning_rate": 0.1, "patch_size": 9, "screen_size": [8, 8],
            "median_kernel": 3, "box": [0, 0, 8, 8], "tv_weight": 0., "seed": 8, "log_every": 10}


class PatchTests(unittest.TestCase):
    def test_median_removes_impulse_and_passes_gradients(self):
        value = torch.ones(1, 1, 5, 5)
        value[0, 0, 2, 2] = 100
        value.requires_grad_()
        filtered = median_pool(value)
        self.assertTrue(torch.equal(filtered, torch.ones_like(filtered)))
        filtered.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_median_invalid_kernel(self):
        for kernel in (0, 2, 11):
            with self.assertRaises(ValueError):
                median_pool(torch.ones(1, 3, 5, 5), kernel)

    def test_patch_does_not_modify_source_or_background(self):
        image = torch.rand(1, 3, 8, 8)
        before = image.clone()
        patch = torch.zeros(1, 3, 3, 3, requires_grad=True)
        result = apply_patch(image, patch, [2, 3, 3, 2])
        expected = before.clone()
        expected[:, :, 3:5, 2:5] = 0
        self.assertTrue(torch.equal(image, before))
        self.assertTrue(torch.equal(result, expected))
        result.sum().backward()
        self.assertGreater(float(patch.grad.abs().sum()), 0)

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            apply_patch(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 2, 2), [7, 7, 2, 2])

    def test_isotropic_tv(self):
        value = torch.tensor([[[[0., 3.], [4., 9.]]]], requires_grad=True)
        self.assertAlmostEqual(float(total_variation(value).detach()), 5.)
        total_variation(value).backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_flat_tv_finite_gradient(self):
        value = torch.ones(1, 3, 4, 4, requires_grad=True)
        total_variation(value).backward()
        self.assertEqual(float(value.grad.abs().sum()), 0.)

    def test_png_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            value = torch.rand(1, 3, 9, 7)
            save_image(value, path)
            self.assertLessEqual(float((load_image(path) - value).abs().max()), 0.5 / 255 + 1e-7)


class AttackTests(unittest.TestCase):
    def test_minimizes_self_similarity(self):
        image = torch.zeros(1, 3, 8, 8)
        image[:, 0] = 1
        original = image.clone()
        result = optimize(toy_models(), image, config())
        self.assertLess(min(r["cosine_sum"] for r in result.history), result.history[0]["cosine_sum"] - 0.2)
        self.assertTrue(torch.equal(image, original))
        self.assertGreaterEqual(float(result.latent_patch.min()), 0.)
        self.assertLessEqual(float(result.latent_patch.max()), 1.)
        screen = render_patch(result.latent_patch, 3, [8, 8])
        actual = cosine_scores(toy_models(), apply_patch(image, screen, [0, 0, 8, 8]), image)
        self.assertAlmostEqual(sum(actual.values()), result.history[result.best_step]["cosine_sum"], places=5)

    def test_seed_reproducibility(self):
        image = torch.rand(1, 3, 8, 8)
        args = {**config(), "steps": 2}
        a, b = optimize(toy_models(), image, args), optimize(toy_models(), image, args)
        self.assertTrue(torch.equal(a.latent_patch, b.latent_patch))
        c = optimize(toy_models(), image, {**args, "seed": 10})
        self.assertFalse(torch.equal(a.latent_patch, c.latent_patch))

    def test_tv_matches_objective(self):
        args = {**config(), "steps": 2, "tv_weight": 0.1}
        result = optimize(toy_models(), torch.rand(1, 3, 8, 8), args)
        for row in result.history:
            self.assertAlmostEqual(row["loss"], row["cosine_sum"] + 0.1 * row["tv"])

    def test_generate_exports_and_evaluate(self):
        import generate
        import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_image(torch.rand(1, 3, 8, 8), root / "input.png")
            cfg = {"canvas_size": [8, 8], "models": [], "attack": {**config(), "steps": 2}}
            (root / "config.json").write_text(json.dumps(cfg))
            argv = ["generate.py", "--image", str(root / "input.png"), "--config", str(root / "config.json"),
                    "--output", str(root / "out"), "--device", "cpu", "--num-patches", "2"]
            with mock_patch("sys.argv", argv), mock_patch("generate.load_models", return_value=(toy_models(), [])), \
                 mock_patch("sys.stdout", new_callable=io.StringIO):
                generate.main()
            for i in range(2):
                output = root / "out" / f"patch_{i:03d}"
                for name in ("patch.png", "latent_patch.png", "patched.png", "patch.pt", "metrics.json", "history.csv"):
                    self.assertTrue((output / name).is_file())
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual(metrics["attack"]["seed"], 8 + i)
                with (output / "history.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
            argv = ["evaluate.py", "--clean", str(root / "out/clean.png"),
                    "--patched", str(root / "out/patch_000/patched.png"),
                    "--config", str(root / "config.json"), "--device", "cpu", "--output", str(root / "eval.json")]
            with mock_patch("sys.argv", argv), mock_patch("evaluate.load_models", return_value=(toy_models(), [])), \
                 mock_patch("sys.stdout", new_callable=io.StringIO):
                evaluate.main()
            evaluated = json.loads((root / "eval.json").read_text())["cosine_similarity"]
            stored = json.loads((root / "out/patch_000/metrics.json").read_text())["saved_png_cosine"]
            for name in evaluated:
                self.assertAlmostEqual(evaluated[name], stored[name], places=6)


class ModelTests(unittest.TestCase):
    def test_prefixes_and_classifier(self):
        expected = {"conv.weight": torch.ones(2, 3), "features.weight": torch.ones(2)}
        for prefix in ("", "module.", "features.module.", "module.features."):
            state = {prefix + key: value for key, value in expected.items()}
            state[prefix + "classifier.weight"] = torch.zeros(9, 2)
            mapped, used, ignored = map_state_dict(state, expected)
            self.assertEqual(set(mapped), set(expected))
            self.assertEqual(used, prefix)
            self.assertIn("classifier.weight", ignored)

    def test_partial_and_wrong_shape_rejected(self):
        expected = {"a": torch.ones(2), "b": torch.ones(1)}
        for state in ({"a": torch.ones(2)}, {"a": torch.ones(3), "b": torch.ones(1)}):
            with self.assertRaises(ValueError):
                map_state_dict(state, expected)

    def test_color_and_normalization(self):
        model = FaceEncoder(nn.Flatten(1), "test", color_order="bgr", mean=(0, 0, 0),
                            std=(1, 1, 1), input_size=(1, 1))
        result = model(torch.tensor([[[[.1]], [[.2]], [[.3]]]]))
        torch.testing.assert_close(result, torch.tensor([[.3, .2, .1]]))

    def test_duplicate_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.pth"
            torch.save({"a": torch.ones(1)}, path)
            specs = [{"name": name, "checkpoint": str(path)} for name in ("arcface", "cosface", "magface")]
            with self.assertRaisesRegex(ValueError, "different"):
                load_models(specs, "cpu")

    def test_backbone_input_gradients(self):
        from dipa.backbones import insightface, magface
        for backend in (insightface, magface):
            model = backend.iresnet18().eval().requires_grad_(False)
            image = torch.randn(1, 3, 112, 112, requires_grad=True)
            embedding = model(image)
            self.assertEqual(embedding.shape, (1, 512))
            embedding.square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(float(image.grad.abs().sum()), 0)
            del model, image, embedding


class UtilityTests(unittest.TestCase):
    def test_alignment(self):
        source = TEMPLATE_112 * 1.5 + [30, 11]
        matrix = similarity_transform(source, TEMPLATE_112)
        recovered = np.column_stack((source, np.ones(5))) @ matrix.T
        np.testing.assert_allclose(recovered[:, :2], TEMPLATE_112, atol=1e-8)

    def test_degenerate_landmarks(self):
        with self.assertRaises(ValueError):
            similarity_transform(np.zeros((5, 2)), TEMPLATE_112)

    def test_camera_detection_failures_are_separate(self):
        base = dict(method="dipa", participant="p1", patch_id="0", expected_identity="p1",
                    clean_identity="p1", clean_detected="1", face_confidence="60")
        rows = [{**base, "trial": "1", "predicted_identity": "", "detected": "0"},
                {**base, "trial": "2", "predicted_identity": "", "detected": "1"},
                {**base, "trial": "3", "predicted_identity": "p1", "detected": "1"},
                {**base, "trial": "4", "predicted_identity": "", "detected": "1", "clean_detected": "0"}]
        stats = summarize(rows)["dipa"]
        self.assertEqual(stats["eligible_trials"], 3)
        self.assertAlmostEqual(stats["identity_change_rate"], 2 / 3)
        self.assertAlmostEqual(stats["detected_dodging_rate"], 1 / 3)
        self.assertAlmostEqual(stats["dodging_rate_given_detection"], 1 / 2)
        self.assertEqual(stats["excluded_clean_failures"], 1)
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])


if __name__ == "__main__":
    unittest.main()
