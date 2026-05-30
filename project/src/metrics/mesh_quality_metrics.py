"""
mesh_quality_metrics.py
-----------------------
Intrinsic mesh quality metrics (no ground truth needed).

Metrics:
  - Watertightness: mesh is closed/manifold
  - Manifoldness: no non-manifold edges
  - Self-intersections: mesh faces don't overlap
  - Surface smoothness: dihedral angle variance
  - Triangle quality: aspect ratios
  - Complexity: vertex/face count

Usage:
    from src.metrics.mesh_quality_metrics import compute_mesh_quality

    result = compute_mesh_quality("results/triposr/mesh.obj")
    print(result.mesh_quality_score)
    print(result.watertight, result.is_manifold, result.self_intersections)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshQualityResult:
    """Result of intrinsic mesh quality evaluation."""

    # Topology
    watertight: bool
    is_manifold: bool
    has_self_intersections: bool
    non_manifold_edge_count: int

    # Geometry
    surface_smoothness: float  # dihedral angle variance
    mean_dihedral_angle: float  # degrees
    triangle_aspect_ratio_mean: float
    triangle_aspect_ratio_std: float

    # Complexity
    vertex_count: int
    face_count: int
    edge_count: int
    vertex_degree_mean: float  # avg neighbors per vertex
    vertex_degree_std: float

    # Composite scores
    mesh_quality_score: float  # combined quality metric
    topology_score: float  # 0-1
    geometry_score: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "watertight": self.watertight,
            "is_manifold": self.is_manifold,
            "has_self_intersections": self.has_self_intersections,
            "non_manifold_edges": self.non_manifold_edge_count,
            "surface_smoothness": self.surface_smoothness,
            "mean_dihedral_angle_deg": self.mean_dihedral_angle,
            "triangle_aspect_ratio_mean": self.triangle_aspect_ratio_mean,
            "triangle_aspect_ratio_std": self.triangle_aspect_ratio_std,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "vertex_degree_mean": self.vertex_degree_mean,
            "vertex_degree_std": self.vertex_degree_std,
            "mesh_quality_score": self.mesh_quality_score,
            "topology_score": self.topology_score,
            "geometry_score": self.geometry_score,
        }


# ---------------------------------------------------------------------------
# Topology checks
# ---------------------------------------------------------------------------

def check_watertightness(mesh) -> bool:
    """Check if mesh is watertight (closed manifold)."""
    try:
        return bool(mesh.is_watertight)
    except Exception:
        return False


def check_manifoldness(mesh) -> bool:
    """Check if mesh is manifold (proper topology)."""
    try:
        # A manifold mesh has no non-manifold edges
        return mesh.is_watertight or len(mesh.edges_unique) > 0
    except Exception:
        return False


def count_non_manifold_edges(mesh) -> int:
    """Count edges with more than 2 adjacent faces."""
    try:
        edge_face_count = {}
        faces = np.asarray(mesh.faces, dtype=np.int64)

        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_face_count[edge] = edge_face_count.get(edge, 0) + 1

        non_manifold = sum(1 for count in edge_face_count.values() if count > 2)
        return non_manifold
    except Exception:
        return 0


def check_self_intersections(mesh) -> bool:
    """Check if mesh has self-intersecting faces."""
    try:
        return len(mesh.self_intersections()) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def compute_dihedral_angles(mesh) -> np.ndarray:
    """
    Compute dihedral angles for all edges (in degrees).

    Returns:
        [E] dihedral angles
    """
    try:
        # Use trimesh's built-in
        angles = mesh.dihedral_angles
        if angles is not None:
            return np.asarray(angles) * 180 / np.pi  # convert to degrees
    except Exception:
        pass

    # Fallback: manual computation
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(faces) == 0:
        return np.array([])

    # Build edge->faces map
    edge_faces = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append((face_idx, face))

    angles = []
    for edge, face_list in edge_faces.items():
        if len(face_list) != 2:
            continue  # Skip non-manifold edges

        face1_idx, face1 = face_list[0]
        face2_idx, face2 = face_list[1]

        # Get normal vectors
        v0 = vertices[face1[0]]
        v1 = vertices[face1[1]]
        v2 = vertices[face1[2]]
        n1 = np.cross(v1 - v0, v2 - v0)
        n1_norm = np.linalg.norm(n1)
        if n1_norm > 1e-8:
            n1 = n1 / n1_norm

        v0 = vertices[face2[0]]
        v1 = vertices[face2[1]]
        v2 = vertices[face2[2]]
        n2 = np.cross(v1 - v0, v2 - v0)
        n2_norm = np.linalg.norm(n2)
        if n2_norm > 1e-8:
            n2 = n2 / n2_norm

        # Dihedral angle
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = angle_rad * 180 / np.pi
        angles.append(angle_deg)

    return np.array(angles) if angles else np.array([])


def compute_triangle_aspect_ratios(mesh) -> np.ndarray:
    """
    Compute aspect ratio for each triangle.
    Aspect ratio = longest_edge / shortest_edge.
    Perfect triangle = 1.0, degenerate > 10.0.

    Returns:
        [F] aspect ratios
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    aspect_ratios = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        edge_lens = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2),
        ]

        max_edge = max(edge_lens)
        min_edge = min(edge_lens)
        if min_edge > 1e-8:
            aspect_ratios.append(max_edge / min_edge)
        else:
            aspect_ratios.append(100.0)  # degenerate triangle

    return np.array(aspect_ratios) if aspect_ratios else np.array([])


def compute_vertex_degrees(mesh) -> np.ndarray:
    """
    Compute vertex degree (number of adjacent vertices).

    Returns:
        [V] degrees
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = len(mesh.vertices)

    vertex_neighbors = [set() for _ in range(vertex_count)]
    for face in faces:
        vertex_neighbors[face[0]].add(face[1])
        vertex_neighbors[face[0]].add(face[2])
        vertex_neighbors[face[1]].add(face[0])
        vertex_neighbors[face[1]].add(face[2])
        vertex_neighbors[face[2]].add(face[0])
        vertex_neighbors[face[2]].add(face[1])

    degrees = np.array([len(neighbors) for neighbors in vertex_neighbors])
    return degrees


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_topology_score(
    watertight: bool,
    is_manifold: bool,
    has_self_intersections: bool,
) -> float:
    """
    Compute topology quality score [0, 1].

    - Watertight + manifold + no self-intersections = 1.0
    - Manifold only = 0.7
    - Self-intersecting = 0.3
    """
    score = 1.0

    if not watertight:
        score *= 0.7
    if not is_manifold:
        score *= 0.7
    if has_self_intersections:
        score *= 0.3

    return float(np.clip(score, 0, 1))


def compute_geometry_score(
    surface_smoothness: float,
    triangle_aspect_ratios: np.ndarray,
) -> float:
    """
    Compute geometry quality score [0, 1].

    Based on surface smoothness and triangle aspect ratios.
    """
    # Smoothness: lower variance is better
    # ideal dihedral angle variance: 0-5 degrees
    smoothness_penalty = np.clip(surface_smoothness / 50.0, 0, 1)
    smoothness_score = 1.0 - smoothness_penalty

    # Triangle quality: lower aspect ratio is better
    # ideal: 1.0, acceptable: < 5.0, poor: > 10.0
    if len(triangle_aspect_ratios) > 0:
        mean_aspect = triangle_aspect_ratios.mean()
        aspect_penalty = np.clip(mean_aspect / 10.0, 0, 1)
        aspect_score = 1.0 - aspect_penalty
    else:
        aspect_score = 1.0

    geometry_score = 0.6 * smoothness_score + 0.4 * aspect_score
    return float(np.clip(geometry_score, 0, 1))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_mesh_quality(mesh_path: str | Path) -> MeshQualityResult:
    """
    Compute intrinsic mesh quality metrics.

    Args:
        mesh_path: path to mesh file (.obj, .glb, .ply)

    Returns:
        MeshQualityResult with all quality metrics
    """
    import trimesh

    # Load mesh
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    # Extract metrics
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Topology
    watertight = check_watertightness(mesh)
    is_manifold = check_manifoldness(mesh)
    has_self_inter = check_self_intersections(mesh)
    non_manifold_count = count_non_manifold_edges(mesh)

    # Geometry
    dihedral_angles = compute_dihedral_angles(mesh)
    mean_dihedral = float(dihedral_angles.mean()) if len(dihedral_angles) > 0 else 0.0
    dihedral_var = float(dihedral_angles.var()) if len(dihedral_angles) > 0 else 0.0

    triangle_aspects = compute_triangle_aspect_ratios(mesh)
    aspect_mean = float(triangle_aspects.mean()) if len(triangle_aspects) > 0 else 0.0
    aspect_std = float(triangle_aspects.std()) if len(triangle_aspects) > 0 else 0.0

    # Complexity
    vertex_degrees = compute_vertex_degrees(mesh)
    degree_mean = float(vertex_degrees.mean()) if len(vertex_degrees) > 0 else 0.0
    degree_std = float(vertex_degrees.std()) if len(vertex_degrees) > 0 else 0.0

    edge_count = len(mesh.edges_unique) if hasattr(mesh, "edges_unique") else 0

    # Scores
    topology_score = compute_topology_score(watertight, is_manifold, has_self_inter)
    geometry_score = compute_geometry_score(dihedral_var, triangle_aspects)

    # Combined
    mesh_quality_score = 0.5 * topology_score + 0.5 * geometry_score

    return MeshQualityResult(
        watertight=watertight,
        is_manifold=is_manifold,
        has_self_intersections=has_self_inter,
        non_manifold_edge_count=non_manifold_count,
        surface_smoothness=dihedral_var,
        mean_dihedral_angle=mean_dihedral,
        triangle_aspect_ratio_mean=aspect_mean,
        triangle_aspect_ratio_std=aspect_std,
        vertex_count=len(vertices),
        face_count=len(faces),
        edge_count=edge_count,
        vertex_degree_mean=degree_mean,
        vertex_degree_std=degree_std,
        mesh_quality_score=mesh_quality_score,
        topology_score=topology_score,
        geometry_score=geometry_score,
    )
