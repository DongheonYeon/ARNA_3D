"""
메시 포맷 변환 모듈

trimesh, pyvista, open3d 간 변환을 담당합니다.
"""

import numpy as np
import trimesh
import pyvista as pv
import open3d as o3d


def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    """
    trimesh를 pyvista PolyData로 변환

    Args:
        mesh: trimesh.Trimesh 객체

    Returns:
        pyvista.PolyData 객체
    """
    vertices = mesh.vertices
    # pyvista faces 형식: [n_verts, v0, v1, v2, ...]
    faces = np.hstack([[3, *f] for f in mesh.faces])
    return pv.PolyData(vertices, faces)


def pyvista_to_trimesh(pv_mesh: pv.PolyData) -> trimesh.Trimesh:
    """
    pyvista PolyData를 trimesh로 변환

    Args:
        pv_mesh: pyvista.PolyData 객체

    Returns:
        trimesh.Trimesh 객체
    """
    vertices = pv_mesh.points
    # pyvista faces: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # n_verts 제거
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    trimesh를 open3d TriangleMesh로 변환

    Args:
        mesh: trimesh.Trimesh 객체

    Returns:
        open3d.geometry.TriangleMesh 객체
    """
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """
    open3d TriangleMesh를 trimesh로 변환

    Args:
        o3d_mesh: open3d.geometry.TriangleMesh 객체

    Returns:
        trimesh.Trimesh 객체
    """
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
    )
