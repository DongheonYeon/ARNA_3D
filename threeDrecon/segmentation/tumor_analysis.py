"""
종양 위치 분석 모듈

NIfTI 볼륨에서 각 종양이 신장 내부에 있는지 외부에 있는지 판별합니다.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label, binary_dilation

from ...config.constants import Label
from ...domain.types import VolumeData


@dataclass
class TumorLocationInfo:
    """종양 위치 정보"""
    tumor_id: int       # 크기순 ID (1, 2, ...)
    is_inside: bool     # 신장 내부 여부
    volume_voxels: int  # 부피 (voxel 수, 크기순 정렬용)


def analyze_tumor_locations(volume: VolumeData) -> list[TumorLocationInfo]:
    """
    각 종양의 신장 내/외부 위치 분석

    판별 기준:
    - is_inside=True: 종양의 모든 voxel이 신장 마스크 내부에 있는 경우
    - is_inside=False: 종양의 일부라도 배경(0)과 닿아있는 경우

    Args:
        volume: 원본 NIfTI 볼륨 데이터

    Returns:
        크기순으로 정렬된 TumorLocationInfo 리스트
    """
    arr = volume.array

    # 종양 및 신장 마스크 추출
    tumor_mask = arr == Label.TUMOR
    kidney_mask = arr == Label.KIDNEY

    if not tumor_mask.any():
        return []

    if not kidney_mask.any():
        # 신장이 없으면 모든 종양은 외부
        return _analyze_without_kidney(tumor_mask)

    # 신장 마스크를 1 voxel dilation (경계 접촉 허용)
    structure = np.ones((3, 3, 3), dtype=bool)
    kidney_dilated = binary_dilation(kidney_mask, structure=structure, iterations=1)

    # 개별 종양 컴포넌트 분리
    labeled, n_tumors = label(tumor_mask, structure=structure)

    results = []
    for i in range(1, n_tumors + 1):
        component = labeled == i
        volume_voxels = int(component.sum())

        # 종양의 모든 voxel이 dilated 신장 마스크 내부에 있는지 확인
        is_inside = bool(np.all(component <= kidney_dilated))

        results.append(TumorLocationInfo(
            tumor_id=i,  # 임시 ID, 정렬 후 재할당
            is_inside=is_inside,
            volume_voxels=volume_voxels,
        ))

    # 크기순 정렬 (extraction.py와 동일한 순서)
    results = sorted(results, key=lambda x: x.volume_voxels, reverse=True)

    # ID 재할당 (1부터 시작)
    for idx, info in enumerate(results, start=1):
        info.tumor_id = idx

    return results


def _analyze_without_kidney(tumor_mask: np.ndarray) -> list[TumorLocationInfo]:
    """신장이 없는 경우 종양 분석 (모두 외부 처리)"""
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_tumors = label(tumor_mask, structure=structure)

    results = []
    for i in range(1, n_tumors + 1):
        component = labeled == i
        volume_voxels = int(component.sum())

        results.append(TumorLocationInfo(
            tumor_id=i,
            is_inside=False,  # 신장이 없으면 모두 외부
            volume_voxels=volume_voxels,
        ))

    # 크기순 정렬
    results = sorted(results, key=lambda x: x.volume_voxels, reverse=True)

    # ID 재할당
    for idx, info in enumerate(results, start=1):
        info.tumor_id = idx

    return results
