from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.reconstruction.triposr_runner import TripoSRConfig, TripoSRCore, sample_points_from_mesh


class FakeRenderer:
    def __init__(self):
        self.chunk_size = None

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size


class FakeMesh:
    def __init__(self):
        self.vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.faces = np.asarray(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            dtype=np.int64,
        )
        self.visual = type(
            "FakeVisual",
            (),
            {
                "vertex_colors": np.asarray(
                    [
                        [255, 0, 0, 255],
                        [0, 255, 0, 255],
                        [0, 0, 255, 255],
                        [255, 255, 255, 255],
                    ],
                    dtype=np.uint8,
                )
            },
        )()

    def export(self, path):
        lines = ["# fake mesh"]
        for vertex in self.vertices:
            lines.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")
        for face in self.faces:
            lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class FakeTripoSRModel:
    def __init__(self):
        self.renderer = FakeRenderer()
        self.device = None
        self.images_seen = None
        self.extract_resolution = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, images, device):
        self.images_seen = images
        self.device = device
        return ["fake-scene-code"]

    def extract_mesh(self, scene_codes, vertex_color, resolution):
        self.extract_resolution = resolution
        return [FakeMesh()]


class TripoSRRunnerTests(unittest.TestCase):
    def test_sample_points_from_mesh_returns_requested_shape(self):
        points = sample_points_from_mesh(FakeMesh(), num_points=32, seed=7, normalize=True)

        self.assertEqual(points.shape, (32, 3))
        self.assertEqual(points.dtype, np.float32)
        self.assertLessEqual(float(np.linalg.norm(points, axis=1).max()), 1.0001)

    def test_core_reconstruct_writes_artifact_contract_with_fake_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "object.png"
            Image.new("RGB", (48, 32), (120, 80, 40)).save(image_path)

            config = TripoSRConfig(
                device="cpu",
                remove_background=False,
                num_points=64,
                mc_resolution=32,
                model_save_format="obj",
            )
            result = TripoSRCore(config=config, model=FakeTripoSRModel()).reconstruct_image(
                image_path=image_path,
                output_dir=temp_path / "outputs",
                name="sample",
                save_preview=False,
            )

            self.assertTrue(result.mesh_path.is_file())
            self.assertIsNotNone(result.colored_mesh_ply_path)
            self.assertTrue(result.colored_mesh_ply_path.is_file())
            self.assertTrue(result.pointcloud_npy_path.is_file())
            self.assertTrue(result.pointcloud_ply_path.is_file())
            self.assertTrue(result.processed_input_path.is_file())
            self.assertTrue(result.summary_path.is_file())
            self.assertEqual(result.points.shape, (64, 3))

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["backend"], "triposr")
            self.assertEqual(summary["num_points"], 64)
            self.assertEqual(summary["mesh"]["format"], "obj")
            self.assertEqual(summary["mesh"]["vertices"], 4)
            self.assertEqual(summary["mesh"]["faces"], 4)
            self.assertEqual(summary["paths"]["mesh"], str(result.mesh_path))

    def test_texture_baking_failure_is_recorded_without_breaking_core_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "object.png"
            Image.new("RGB", (48, 32), (120, 80, 40)).save(image_path)

            config = TripoSRConfig(
                device="cpu",
                remove_background=False,
                num_points=64,
                mc_resolution=32,
                model_save_format="obj",
                bake_texture=True,
                texture_resolution=32,
            )
            result = TripoSRCore(config=config, model=FakeTripoSRModel()).reconstruct_image(
                image_path=image_path,
                output_dir=temp_path / "outputs",
                name="sample",
                save_preview=False,
            )

            self.assertTrue(result.mesh_path.is_file())
            self.assertTrue(result.pointcloud_ply_path.is_file())
            self.assertIsNone(result.textured_mesh_obj_path)
            self.assertIsNone(result.texture_path)

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["texture_baking"]["enabled"])
            self.assertFalse(summary["texture_baking"]["success"])
            self.assertIsInstance(summary["texture_baking"]["error"], str)


if __name__ == "__main__":
    unittest.main()
