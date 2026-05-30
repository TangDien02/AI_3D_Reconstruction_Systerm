from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class TextureBakingError(RuntimeError):
    """Raised when textured mesh baking cannot run in the current environment."""


@dataclass(frozen=True)
class BakedTexture:
    vmapping: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    colors: np.ndarray


@dataclass(frozen=True)
class TexturedMeshExport:
    obj_path: Path
    mtl_path: Path
    texture_path: Path


def _require_runtime():
    try:
        import moderngl
        import trimesh
        import xatlas
    except Exception as exc:
        raise TextureBakingError(
            "Texture baking requires xatlas, moderngl, and trimesh. "
            "Install project/requirements-triposr.txt and ensure OpenGL/EGL is available."
        ) from exc
    return moderngl, trimesh, xatlas


def make_atlas(mesh: Any, texture_resolution: int, texture_padding: int) -> dict[str, np.ndarray]:
    _, _, xatlas = _require_runtime()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise TextureBakingError("Mesh vertices must have shape [N, 3].")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise TextureBakingError("Mesh faces must have shape [M, 3].")

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    options = xatlas.PackOptions()
    options.resolution = int(texture_resolution)
    options.padding = int(texture_padding)
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": np.asarray(vmapping, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "uvs": np.asarray(uvs, dtype=np.float32),
    }


def rasterize_position_atlas(
    mesh: Any,
    atlas_vmapping: np.ndarray,
    atlas_indices: np.ndarray,
    atlas_uvs: np.ndarray,
    texture_resolution: int,
    texture_padding: int,
) -> np.ndarray:
    moderngl, _, _ = _require_runtime()
    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception as exc:
        raise TextureBakingError(
            "Could not create a standalone OpenGL context for texture baking."
        ) from exc

    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 v_pos;
            void main() {
                v_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(v_pos, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            out vec3 vg_pos;
            void main() {
                vg_pos = in_pos;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        geometry_shader="""
            #version 330
            uniform float u_resolution;
            uniform float u_dilation;
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 12) out;
            in vec3 vg_pos[];
            out vec3 vf_pos;
            void lineSegment(int aidx, int bidx) {
                vec2 a = gl_in[aidx].gl_Position.xy;
                vec2 b = gl_in[bidx].gl_Position.xy;
                vec3 aCol = vg_pos[aidx];
                vec3 bCol = vg_pos[bidx];

                vec2 dir = normalize((b - a) * u_resolution);
                vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                gl_Position = vec4(a + offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(a - offset, 0.0, 1.0);
                vf_pos = aCol;
                EmitVertex();
                gl_Position = vec4(b + offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
                gl_Position = vec4(b - offset, 0.0, 1.0);
                vf_pos = bCol;
                EmitVertex();
            }
            void main() {
                lineSegment(0, 1);
                lineSegment(1, 2);
                lineSegment(2, 0);
                EndPrimitive();
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 vf_pos;
            out vec4 o_col;
            void main() {
                o_col = vec4(vf_pos, 1.0);
            }
        """,
    )

    uvs = np.asarray(atlas_uvs, dtype=np.float32).reshape(-1, 2).flatten()
    pos = np.asarray(mesh.vertices, dtype=np.float32)[atlas_vmapping].reshape(-1, 3).flatten()
    indices = np.asarray(atlas_indices, dtype=np.int32).flatten()
    vbo_uvs = ctx.buffer(uvs)
    vbo_pos = ctx.buffer(pos)
    ibo = ctx.buffer(indices)
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((texture_resolution, texture_resolution), 4, dtype="f4")]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    gs_prog["u_resolution"].value = float(texture_resolution)
    gs_prog["u_dilation"].value = float(texture_padding)
    gs_vao.render()
    basic_vao.render()

    fbo_bytes = fbo.color_attachments[0].read()
    return np.frombuffer(fbo_bytes, dtype="f4").reshape(texture_resolution, texture_resolution, 4)


def positions_to_colors(
    model: Any,
    scene_code: Any,
    positions_texture: np.ndarray,
    texture_resolution: int,
) -> np.ndarray:
    if not hasattr(model, "renderer") or not hasattr(model.renderer, "query_triplane"):
        raise TextureBakingError("Model renderer does not expose query_triplane.")
    if not hasattr(model, "decoder"):
        raise TextureBakingError("Model does not expose decoder.")

    scene_device = getattr(scene_code, "device", None)
    if scene_device is None:
        scene_device = getattr(model, "device", "cpu")
    positions = torch.as_tensor(
        positions_texture.reshape(-1, 4)[:, :-1],
        device=scene_device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(model.decoder, positions, scene_code)
    rgb = queried_grid["color"].detach().cpu().numpy().reshape(-1, 3)
    alpha = positions_texture.reshape(-1, 4)[:, -1]
    rgba = np.insert(rgb, 3, alpha, axis=1)
    rgba[rgba[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba.reshape(texture_resolution, texture_resolution, 4).astype(np.float32)


def bake_texture(
    mesh: Any,
    model: Any,
    scene_code: Any,
    texture_resolution: int = 1024,
    texture_padding: int | None = None,
) -> BakedTexture:
    texture_resolution = int(texture_resolution)
    if texture_resolution <= 0:
        raise ValueError("texture_resolution must be greater than 0.")
    texture_padding = int(texture_padding or round(max(2, texture_resolution / 256)))

    atlas = make_atlas(mesh, texture_resolution, texture_padding)
    positions_texture = rasterize_position_atlas(
        mesh,
        atlas["vmapping"],
        atlas["indices"],
        atlas["uvs"],
        texture_resolution,
        texture_padding,
    )
    colors_texture = positions_to_colors(model, scene_code, positions_texture, texture_resolution)
    return BakedTexture(
        vmapping=atlas["vmapping"],
        indices=atlas["indices"],
        uvs=atlas["uvs"],
        colors=colors_texture,
    )


def _texture_to_uint8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32)
    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def export_textured_obj(
    mesh: Any,
    baked_texture: BakedTexture,
    output_path: str | Path,
    texture_name: str = "texture.png",
) -> TexturedMeshExport:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(baked_texture.indices, dtype=np.int64).reshape(-1, 3)
    vmapping = np.asarray(baked_texture.vmapping, dtype=np.int64)
    uvs = np.asarray(baked_texture.uvs, dtype=np.float32).reshape(-1, 2)
    if len(vertices) == 0 or len(faces) == 0 or len(vmapping) != len(uvs):
        raise TextureBakingError("Baked texture atlas has an invalid OBJ contract.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_path.with_suffix(".mtl")
    texture_path = output_path.with_name(texture_name)

    Image.fromarray(_texture_to_uint8(baked_texture.colors), mode="RGBA").save(texture_path)
    material_name = output_path.stem
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        "# Textured mesh generated by TripoSR texture baking",
        f"mtllib {mtl_path.name}",
        f"usemtl {material_name}",
    ]
    for atlas_vertex_idx in vmapping:
        x, y, z = vertices[int(atlas_vertex_idx)]
        lines.append(f"v {x:.7f} {y:.7f} {z:.7f}")
    for u, v in uvs:
        lines.append(f"vt {u:.7f} {1.0 - v:.7f}")
    for a, b, c in faces:
        ia, ib, ic = int(a) + 1, int(b) + 1, int(c) + 1
        lines.append(f"f {ia}/{ia} {ib}/{ib} {ic}/{ic}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TexturedMeshExport(obj_path=output_path, mtl_path=mtl_path, texture_path=texture_path)
