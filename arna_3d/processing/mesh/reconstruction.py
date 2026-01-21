"""
Poisson 재구성 모듈

분리된 혈관 메시를 Poisson 재구성으로 병합합니다.
"""

import numpy as np
import trimesh
import open3d as o3d

from ...config.constants import PoissonParams
from ...core.types import MeshCollection
from .conversion import trimesh_to_open3d, open3d_to_trimesh


def poisson_reconstruct(
    meshes: list[trimesh.Trimesh],
    depth: int = PoissonParams.DEPTH,
    sample_points: int = PoissonParams.SAMPLE_POINTS,
) -> trimesh.Trimesh:
    """
    여러 메시를 Poisson 재구성으로 병합

    Args:
        meshes: 병합할 메시 리스트
        depth: Poisson 재구성 깊이
        sample_points: 포인트 샘플링 수

    Returns:
        재구성된 메시
    """
    if not meshes:
        raise ValueError("[ERROR] Input mesh list is empty")

    print(f"{'':7}- Merged mesh: {len(meshes)}")

    # 메시 병합
    merged = trimesh.util.concatenate(meshes)

    # Open3D로 변환
    o3d_mesh = trimesh_to_open3d(merged)

    # 포인트 샘플링
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=sample_points)
    pcd.estimate_normals()

    # Poisson 재구성
    mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh_out = mesh_out.crop(pcd.get_axis_aligned_bounding_box())

    # trimesh로 변환
    tri_mesh = open3d_to_trimesh(mesh_out)

    # 가장 큰 연결 요소만 유지
    components = tri_mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"{'':7}- Connected components: {len(components)} - using largest")
        tri_mesh = max(components, key=lambda c: c.area)

    return tri_mesh


def process_vessel_reconstruction(collection: MeshCollection) -> MeshCollection:
    """
    혈관 그룹을 Poisson 재구성으로 처리

    Artery + Renal_a → Artery
    Vein + Renal_v → Vein

    Args:
        collection: 입력 MeshCollection

    Returns:
        재구성된 MeshCollection
    """
    # 병합 대상 그룹 정의
    artery_group = ["Artery", "Renal_a"]
    vein_group = ["Vein", "Renal_v"]

    result = MeshCollection()

    # Artery 그룹 처리
    artery_meshes = [
        mesh for name, mesh in collection.items()
        if any(name == g or name.startswith(f"{g}-") for g in artery_group)
    ]
    if artery_meshes:
        print("[INFO] Processing Artery group")
        reconstructed_artery = poisson_reconstruct(artery_meshes)
        result.add("Artery", reconstructed_artery)

    # Vein 그룹 처리
    vein_meshes = [
        mesh for name, mesh in collection.items()
        if any(name == g or name.startswith(f"{g}-") for g in vein_group)
    ]
    if vein_meshes:
        print("[INFO] Processing Vein group")
        reconstructed_vein = poisson_reconstruct(vein_meshes)
        result.add("Vein", reconstructed_vein)

    # 나머지 구조물은 그대로 추가
    excluded = set(artery_group + vein_group)
    for name, mesh in collection.items():
        # 이미 처리된 그룹에 속하지 않는 것만 추가
        base_name = name.split("-")[0]
        if base_name not in excluded:
            result.add(name, mesh)

    return result
