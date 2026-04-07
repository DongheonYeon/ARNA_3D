"""
Poisson 재구성 모듈

분리된 혈관 메시를 Poisson 재구성으로 병합합니다.
"""

import numpy as np
import trimesh
import open3d as o3d

from ...config.constants import PoissonParams
from ...config.logger import logger
from ...domain.types import MeshCollection
from .conversion import trimesh_to_open3d, open3d_to_trimesh


def _create_bbox_mesh(
    bbox: o3d.geometry.AxisAlignedBoundingBox,
    color: tuple[int, int, int, int] = (255, 255, 0, 128),
) -> trimesh.Trimesh:
    """
    Open3D AxisAlignedBoundingBox를 solid box 메시로 변환

    Args:
        bbox: Open3D bounding box
        color: RGBA 색상 (기본값: 반투명 노란색)

    Returns:
        solid box 메시
    """
    min_bound = np.array(bbox.min_bound)
    max_bound = np.array(bbox.max_bound)

    # 박스 크기 및 중심 계산
    extents = max_bound - min_bound
    center = (min_bound + max_bound) / 2

    # solid box 생성
    box_mesh = trimesh.creation.box(extents=extents)
    box_mesh.apply_translation(center)

    # GLB 호환 PBR material 적용
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255],
        metallicFactor=0.0,
        roughnessFactor=0.5,
    )
    box_mesh.visual = trimesh.visual.TextureVisuals(material=material)

    return box_mesh


def poisson_reconstruct(
    meshes: list[trimesh.Trimesh],
    depth: int = PoissonParams.DEPTH,
    sample_points: int = PoissonParams.SAMPLE_POINTS,
    use_lcc: bool = True,
    return_bbox: bool = False,
) -> trimesh.Trimesh | tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    여러 메시를 Poisson 재구성으로 병합

    Args:
        meshes: 병합할 메시 리스트
        depth: Poisson 재구성 깊이
        sample_points: 포인트 샘플링 수
        use_lcc: True면 가장 큰 연결 요소만 유지
        return_bbox: True면 crop bbox 메시도 함께 반환

    Returns:
        재구성된 메시 또는 (재구성 메시, bbox 메시)
    """
    if not meshes:
        raise ValueError("[ERROR] Input mesh list is empty")

    logger.debug(f"       - Merged mesh: {len(meshes)}")

    # 메시 병합
    merged = trimesh.util.concatenate(meshes)

    # Open3D로 변환
    o3d_mesh = trimesh_to_open3d(merged)

    # 포인트 샘플링
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=sample_points)
    pcd.estimate_normals()

    # Poisson 재구성
    mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    bbox = pcd.get_axis_aligned_bounding_box()
    '''
            y(+)
    [max]   ↑
     x(+) ← +--------+
           /        /|
          /        / |
         +-------+|  |
         |        |  |
         |        |  +
         |        | /
         |        |/
         +--------+ → x(+)
                ↙ ↓   [min]
            z(+) y(+)
    '''
    min_padding = np.array([10.0, 0.0, 10.0])
    max_padding = np.array([10.0, 0.0, 10.0])
    padded_min = np.array(bbox.min_bound) - min_padding
    padded_max = np.array(bbox.max_bound) + max_padding
    crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=padded_min, max_bound=padded_max)
    mesh_out = mesh_out.crop(crop_bbox)

    # trimesh로 변환
    tri_mesh = open3d_to_trimesh(mesh_out)

    # 가장 큰 연결 요소만 유지 (옵션)
    if use_lcc:
        components = tri_mesh.split(only_watertight=False)
        if len(components) > 1:
            logger.debug(f"       - Connected components: {len(components)} - using largest")
            tri_mesh = max(components, key=lambda c: c.area)

    if return_bbox:
        # 실제 crop에 사용된 bbox (패딩 적용됨) 시각화
        bbox_mesh = _create_bbox_mesh(crop_bbox)
        return tri_mesh, bbox_mesh

    return tri_mesh


def process_vessel_reconstruction(
    collection: MeshCollection,
    use_lcc: bool = False,
    debug: bool = False,
) -> MeshCollection:
    """
    혈관 그룹을 Poisson 재구성으로 처리

    Artery + Renal_a -> Artery
    Vein + Renal_v -> Vein

    Args:
        collection: 입력 MeshCollection
        use_lcc: True면 재구성 후 가장 큰 연결 요소만 유지 (기본값: False)
        debug: True면 crop에 사용된 bounding box 메시도 추가

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
        logger.debug("Processing Artery group")
        if debug:
            reconstructed_artery, artery_bbox = poisson_reconstruct(
                artery_meshes, use_lcc=use_lcc, return_bbox=True
            )
            result.add("Artery", reconstructed_artery)
            result.add("Artery_bbox", artery_bbox)
        else:
            reconstructed_artery = poisson_reconstruct(artery_meshes, use_lcc=use_lcc)
            result.add("Artery", reconstructed_artery)

    # Vein 그룹 처리
    vein_meshes = [
        mesh for name, mesh in collection.items()
        if any(name == g or name.startswith(f"{g}-") for g in vein_group)
    ]
    if vein_meshes:
        logger.debug("Processing Vein group")
        if debug:
            reconstructed_vein, vein_bbox = poisson_reconstruct(
                vein_meshes, use_lcc=use_lcc, return_bbox=True
            )
            result.add("Vein", reconstructed_vein)
            result.add("Vein_bbox", vein_bbox)
        else:
            reconstructed_vein = poisson_reconstruct(vein_meshes, use_lcc=use_lcc)
            result.add("Vein", reconstructed_vein)

    # 나머지 구조물은 그대로 추가
    excluded = set(artery_group + vein_group)
    for name, mesh in collection.items():
        # 이미 처리된 그룹에 속하지 않는 것만 추가
        base_name = name.split("-")[0]
        if base_name not in excluded:
            result.add(name, mesh)

    return result
