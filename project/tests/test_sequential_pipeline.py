from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from test_triposr_runner import FakeTripoSRModel
from src.pipeline.sequential_3d_pipeline import (
    Sequential3DPipeline,
    Sequential3DPipelineConfig,
)
from src.reconstruction.triposr_runner import TripoSRConfig, TripoSRCore


class Sequential3DPipelineTests(unittest.TestCase):
    def test_pipeline_runs_stages_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "concept.png"
            Image.new("RGB", (48, 32), (120, 80, 40)).save(image_path)

            triposr_config = TripoSRConfig(
                device="cpu",
                remove_background=False,
                num_points=32,
                mc_resolution=16,
                model_save_format="obj",
            )
            core = TripoSRCore(config=triposr_config, model=FakeTripoSRModel())
            pipeline = Sequential3DPipeline(
                config=Sequential3DPipelineConfig(
                    output_dir=str(temp_path / "outputs"),
                    save_preview=False,
                    triposr=triposr_config,
                ),
                triposr_core=core,
            )

            result = pipeline.run_image(image_path, job_id="concept_art")

            self.assertEqual(result.status, "done")
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual([stage.name for stage in result.stages], ["validate_input", "reconstruct_triposr"])
            self.assertTrue(result.reconstruction.mesh_path.is_file())

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["job_id"], "concept_art")
            self.assertEqual(manifest["status"], "done")
            self.assertEqual(manifest["backend"], "triposr")
            self.assertTrue(manifest["artifacts"]["mesh"].endswith("mesh.obj"))
            self.assertEqual(len(manifest["stages"]), 2)


if __name__ == "__main__":
    unittest.main()
