"""
메시 변환 모듈

회전, 이동 등 기하학적 변환을 담당합니다.
"""

import numpy as np
import trimesh


def rotate_and_center_scene(scene: trimesh.Scene) -> trimesh.Scene:
    """
    Scene을 회전하고 중심으로 이동

    NIfTI 좌표계 -> GLB 좌표계 변환을 수행합니다.

    Args:
        scene: 변환할 Scene

    Returns:
        변환된 새 Scene
    """
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2

    # 중심 이동 행렬
    T = trimesh.transformations.translation_matrix(-center)

    # 회전 행렬 (X축 기준 -90도)
    angle_rad = np.pi / 2
    R = trimesh.transformations.rotation_matrix(angle_rad, direction=[-1, 0, 0], point=center)

    # 변환 적용
    combined_transform = trimesh.transformations.concatenate_matrices(T, R)
    scene.apply_transform(combined_transform)

    # 변환된 지오메트리로 새 Scene 생성
    transformed_geometries = scene.dump()
    return trimesh.Scene(transformed_geometries)


def rotate_and_center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    단일 메시를 회전하고 중심으로 이동

    Args:
        mesh: 변환할 메시

    Returns:
        변환된 메시
    """
    center = mesh.centroid

    # 중심 이동 행렬
    T = trimesh.transformations.translation_matrix(-center)

    # 회전 행렬 (X축 기준 -90도)
    angle_rad = np.pi / 2
    R = trimesh.transformations.rotation_matrix(angle_rad, direction=[-1, 0, 0], point=center)

    # 변환 적용
    combined_transform = trimesh.transformations.concatenate_matrices(T, R)
    mesh.apply_transform(combined_transform)

    return mesh
