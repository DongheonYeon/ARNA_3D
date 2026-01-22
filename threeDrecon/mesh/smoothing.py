"""
메시 스무딩 모듈

Taubin/Laplacian 스무딩, Dilation, Decimation을 수행합니다.
"""

import pyvista as pv
import trimesh

from ...config.settings import SmoothingConfig
from ...config.logger import logger
from ...domain.types import MeshCollection
from .conversion import trimesh_to_pyvista, pyvista_to_trimesh


# 스무딩 함수 맵
SMOOTHING_FUNC_MAP = {
    "laplacian": pv.PolyData.smooth,
    "taubin": pv.PolyData.smooth_taubin,
    None: lambda mesh, **kwargs: mesh,
}

# Simplification 함수 맵
SIMPLIFICATION_FUNC_MAP = {
    "decimate": pv.PolyData.decimate,
    "decimate_pro": pv.PolyData.decimate_pro,
    None: lambda mesh, **kwargs: mesh,
}


def apply_dilation(pv_mesh: pv.PolyData, offset: float = 0.5) -> pv.PolyData:
    """
    메시 Dilation (노멀 방향으로 확장)

    Args:
        pv_mesh: pyvista PolyData
        offset: 확장 거리

    Returns:
        확장된 메시
    """
    pv_mesh.compute_normals(inplace=True)
    pv_mesh.points += offset * pv_mesh.point_normals
    return pv_mesh


def apply_smoothing(
    pv_mesh: pv.PolyData,
    method: str | None,
    **kwargs,
) -> pv.PolyData:
    """
    메시 스무딩 적용

    Args:
        pv_mesh: pyvista PolyData
        method: 스무딩 방법 ('taubin', 'laplacian', None)
        **kwargs: 스무딩 파라미터

    Returns:
        스무딩된 메시
    """
    func = SMOOTHING_FUNC_MAP.get(method)
    if func is None:
        raise ValueError(f"알 수 없는 스무딩 방법: {method}")

    return func(pv_mesh, **kwargs)


def apply_simplification(
    pv_mesh: pv.PolyData,
    method: str | None,
    **kwargs,
) -> pv.PolyData:
    """
    메시 Simplification (Decimation) 적용

    Args:
        pv_mesh: pyvista PolyData
        method: Simplification 방법 ('decimate', 'decimate_pro', None)
        **kwargs: Simplification 파라미터

    Returns:
        간소화된 메시
    """
    func = SIMPLIFICATION_FUNC_MAP.get(method)
    if func is None:
        logger.warning(f"Unknown simplification method: {method}")
        return pv_mesh

    return func(pv_mesh, **kwargs)


def process_single_mesh(
    mesh: trimesh.Trimesh,
    config: SmoothingConfig,
) -> trimesh.Trimesh:
    """
    단일 메시에 스무딩 설정 적용

    Args:
        mesh: 처리할 메시
        config: 스무딩 설정

    Returns:
        처리된 메시
    """
    logger.debug(f"Processing {config.name}")
    pv_mesh = trimesh_to_pyvista(mesh)

    # Dilation
    if config.dilation_func:
        logger.debug(f"       - Apply Dilation")
        if config.dilation_func == "default":
            pv_mesh = apply_dilation(pv_mesh, **config.dilation_kwargs)
        else:
            raise ValueError(f"알 수 없는 dilation_func: {config.dilation_func}")

    # Smoothing
    if config.smoothing_func:
        logger.debug(f"       - Apply Smoothing")
        pv_mesh = apply_smoothing(pv_mesh, config.smoothing_func, **config.smoothing_kwargs)

    # Simplification
    if config.simplification_func:
        logger.debug(f"       - Apply Simplification ({config.simplification_func})")
        pv_mesh = apply_simplification(pv_mesh, config.simplification_func, **config.simplification_kwargs)

    return pyvista_to_trimesh(pv_mesh)


def smooth_mesh_collection(
    collection: MeshCollection,
    configs: list[SmoothingConfig],
) -> MeshCollection:
    """
    MeshCollection의 모든 메시에 스무딩 적용

    Args:
        collection: 처리할 MeshCollection
        configs: 각 파트별 스무딩 설정 리스트

    Returns:
        처리된 새 MeshCollection
    """
    result = MeshCollection()

    for config in configs:
        # 설정 이름과 일치하거나 접두사로 시작하는 메시 찾기
        matching = collection.get_by_prefix(config.name)

        if not matching:
            logger.warning(f"Skip {config.name}: mesh not found in collection.")
            continue

        for mesh_name, mesh in matching:
            # 설정 복사 후 이름 변경
            mesh_config = SmoothingConfig(
                name=mesh_name,
                smoothing_func=config.smoothing_func,
                smoothing_kwargs=config.smoothing_kwargs,
                dilation_func=config.dilation_func,
                dilation_kwargs=config.dilation_kwargs,
                simplification_func=config.simplification_func,
                simplification_kwargs=config.simplification_kwargs,
            )

            processed = process_single_mesh(mesh, mesh_config)
            result.add(mesh_name, processed)

    return result
