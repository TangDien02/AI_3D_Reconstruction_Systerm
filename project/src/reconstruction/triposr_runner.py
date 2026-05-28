from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TripoSRConfig:
    triposr_dir: Path | None
    python_executable: str
    device: str
    pretrained_model: str
    chunk_size: int
    mc_resolution: int
    model_save_format: str
    bake_texture: bool
    texture_resolution: int
    no_remove_bg: bool
    foreground_ratio: float
    render: bool
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "TripoSRConfig":
        triposr_dir_value = os.environ.get("TRIPOSR_DIR")
        triposr_dir = Path(triposr_dir_value).expanduser() if triposr_dir_value else None
        model_save_format = os.environ.get("TRIPOSR_MODEL_SAVE_FORMAT", "glb").strip().lower()
        if model_save_format not in {"glb", "obj"}:
            model_save_format = "glb"
        return cls(
            triposr_dir=triposr_dir,
            python_executable=os.environ.get("TRIPOSR_PYTHON", sys.executable),
            device=os.environ.get("TRIPOSR_DEVICE", "cuda:0"),
            pretrained_model=os.environ.get("TRIPOSR_PRETRAINED_MODEL", "stabilityai/TripoSR"),
            chunk_size=_env_int("TRIPOSR_CHUNK_SIZE", 8192),
            mc_resolution=_env_int("TRIPOSR_MC_RESOLUTION", 256),
            model_save_format=model_save_format,
            bake_texture=_env_bool("TRIPOSR_BAKE_TEXTURE", False),
            texture_resolution=_env_int("TRIPOSR_TEXTURE_RESOLUTION", 2048),
            no_remove_bg=_env_bool("TRIPOSR_NO_REMOVE_BG", False),
            foreground_ratio=_env_float("TRIPOSR_FOREGROUND_RATIO", 0.85),
            render=_env_bool("TRIPOSR_RENDER", False),
            timeout_seconds=_env_int("TRIPOSR_TIMEOUT_SECONDS", 600),
        )

    @property
    def run_script(self) -> Path | None:
        if self.triposr_dir is None:
            return None
        return self.triposr_dir / "run.py"

    def readiness(self) -> dict[str, object]:
        run_script = self.run_script
        return {
            "backend": "triposr",
            "available": bool(run_script and run_script.is_file()),
            "triposr_dir": str(self.triposr_dir) if self.triposr_dir else None,
            "run_script": str(run_script) if run_script else None,
            "python_executable": self.python_executable,
            "device": self.device,
            "pretrained_model": self.pretrained_model,
            "model_save_format": self.model_save_format,
            "bake_texture": self.bake_texture,
            "texture_resolution": self.texture_resolution,
            "no_remove_bg": self.no_remove_bg,
            "foreground_ratio": self.foreground_ratio,
            "mc_resolution": self.mc_resolution,
            "chunk_size": self.chunk_size,
            "timeout_seconds": self.timeout_seconds,
        }


class TripoSRRunner:
    def __init__(self, config: TripoSRConfig | None = None):
        self.config = config or TripoSRConfig.from_env()

    def validate_ready(self) -> None:
        run_script = self.config.run_script
        if run_script is None:
            raise RuntimeError(
                "TRIPOSR_DIR is not configured. Set it to a local checkout of "
                "https://github.com/VAST-AI-Research/TripoSR."
            )
        if not run_script.is_file():
            raise RuntimeError(f"TripoSR run.py not found: {run_script}")

    def build_command(self, image_path: Path, output_dir: Path) -> list[str]:
        self.validate_ready()
        command = [
            self.config.python_executable,
            str(self.config.run_script),
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--model-save-format",
            self.config.model_save_format,
            "--device",
            self.config.device,
            "--pretrained-model-name-or-path",
            self.config.pretrained_model,
            "--chunk-size",
            str(self.config.chunk_size),
            "--mc-resolution",
            str(self.config.mc_resolution),
            "--foreground-ratio",
            str(self.config.foreground_ratio),
        ]
        if self.config.bake_texture:
            command.extend(["--bake-texture", "--texture-resolution", str(self.config.texture_resolution)])
        if self.config.no_remove_bg:
            command.append("--no-remove-bg")
        if self.config.render:
            command.append("--render")
        return command

    def reconstruct(self, image_path: str | Path, output_dir: str | Path) -> dict[str, object]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        raw_output_dir = output_dir / "triposr_raw"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(image_path=image_path, output_dir=raw_output_dir)
        started_at = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(self.config.triposr_dir),
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or completed.stdout or "").strip()[-3000:]
            raise RuntimeError(f"TripoSR failed with exit code {completed.returncode}: {stderr_tail}")

        mesh_path = raw_output_dir / "0" / f"mesh.{self.config.model_save_format}"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"TripoSR did not produce expected mesh: {mesh_path}")

        texture_path = raw_output_dir / "0" / "texture.png"
        render_path = raw_output_dir / "0" / "render.mp4"
        prepared_input_path = raw_output_dir / "0" / "input.png"

        return {
            "backend": "triposr",
            "primary_output": "mesh_glb" if self.config.model_save_format == "glb" else "mesh_obj",
            "mesh_path": str(mesh_path),
            "texture_path": str(texture_path) if texture_path.is_file() else None,
            "render_path": str(render_path) if render_path.is_file() else None,
            "prepared_input_path": str(prepared_input_path) if prepared_input_path.is_file() else None,
            "raw_output_dir": str(raw_output_dir),
            "processing_ms": elapsed_ms,
            "command": command,
            "stdout_tail": (completed.stdout or "").strip()[-3000:],
            "stderr_tail": (completed.stderr or "").strip()[-3000:],
            "config": self.config.readiness(),
        }

