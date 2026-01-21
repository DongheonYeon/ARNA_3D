"""
메시 분할 모듈

메시를 좌/우로 분할하거나 유효한 부분만 필터링합니다.
"""

import trimesh

from config.constants import MorphologyParams
from config.logger import logger


def split_bilateral(
    mesh: trimesh.Trimesh,
    max_parts: int = MorphologyParams.MAX_BILATERAL_PARTS,
) -> list[trimesh.Trimesh]:
    """
    메시를 좌/우로 분할 (Z축 기준 정렬)

    Args:
        mesh: 분할할 메시
        max_parts: 최대 파트 개수

    Returns:
        분할된 메시 리스트 (Z축 기준 정렬됨)
    """
    parts = mesh.split(only_watertight=False)
    if not parts:
        return []

    # 크기순으로 상위 max_parts개 선택
    parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)[:max_parts]

    # Z축 (centroid[2]) 기준 정렬 (L → R 순서)
    parts = sorted(parts, key=lambda m: m.centroid[2])

    return parts


def filter_valid_tumors(meshes: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """
    유효한 종양 메시만 필터링

    내부 빈 공간 메시(음수 볼륨)를 제거합니다.

    Args:
        meshes: 종양 메시 리스트

    Returns:
        유효한 메시만 포함된 리스트
    """
    valid_parts = []

    for part in meshes:
        # 볼륨이 음수면 내부 공간 (노멀이 안쪽을 향함)
        if part.is_watertight and part.volume < 0:
            logger.debug(f"Removing tumor cavity (negative volume: {part.volume:.2f})")
            continue
        valid_parts.append(part)

    return valid_parts
