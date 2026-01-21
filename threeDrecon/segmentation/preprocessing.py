"""
세그멘테이션 전처리 모듈

NIfTI 세그멘테이션 라벨을 전처리합니다.
"""

import numpy as np
from scipy.ndimage import binary_dilation

from config.constants import Label, MorphologyParams
from domain.types import VolumeData
from file_io.nifti import copy_metadata
from threeDrecon.vessel.branch import process_vessel_branches


def apply_fat_dilation(
    label_array: np.ndarray,
    iterations: int = MorphologyParams.FAT_DILATION_ITERATIONS,
) -> np.ndarray:
    """
    Fat 영역을 신장 경계까지 확장

    혈관, 요관, 종양 영역은 제외합니다.

    Args:
        label_array: 세그멘테이션 라벨 배열
        iterations: dilation 반복 횟수

    Returns:
        확장된 Fat 마스크
    """
    tumor_mask = label_array == Label.TUMOR
    kidney_mask = label_array == Label.KIDNEY
    fat_mask = label_array == Label.FAT

    structure = np.ones((3, 3, 3), dtype=bool)
    dilated_kidney = binary_dilation(kidney_mask, structure=structure, iterations=iterations)

    # 신장 경계 영역 계산
    kidney_boundary = dilated_kidney & ~kidney_mask
    kidney_boundary[tumor_mask] = False  # 종양 영역 제외

    return fat_mask | kidney_boundary


def get_kidney_z_range(label_array: np.ndarray) -> tuple[int, int]:
    """
    신장이 존재하는 Z 범위 반환

    Args:
        label_array: 세그멘테이션 라벨 배열

    Returns:
        (시작 Z, 끝 Z)
    """
    kidney = (label_array == Label.KIDNEY).any(axis=(1, 2))
    indices = np.where(kidney)[0]
    return int(indices[0]), int(indices[-1])


def preprocess_segmentation(volume: VolumeData) -> VolumeData:
    """
    세그멘테이션 전처리 메인 함수

    1. 혈관 분기 자동 분할 (Renal_a, Renal_v 생성)
    2. Fat dilation 적용

    Args:
        volume: 입력 VolumeData

    Returns:
        전처리된 VolumeData
    """
    label_array = volume.array

    # 신장 Z 범위 계산
    z_start, z_end = get_kidney_z_range(label_array)

    # 혈관 분기 처리
    renal_a, renal_v = process_vessel_branches(label_array, z_start, z_end)

    # 혈관 마스크 생성 (원본 + 분기)
    vessel_mask = (
        (label_array == Label.ARTERY)
        | (label_array == Label.VEIN)
        | renal_a.astype(bool)
        | renal_v.astype(bool)
    )

    # Fat dilation 적용
    fat_mask_dilated = apply_fat_dilation(label_array)

    # 결과 라벨 배열 생성
    out_arr = label_array.copy()

    # Fat 라벨 추가 (확장된 영역)
    out_arr[fat_mask_dilated] = Label.FAT

    # 원본 라벨 복원 (우선순위 높은 것들)
    ureter_mask = label_array == Label.URETER
    out_arr[vessel_mask] = label_array[vessel_mask]  # 원본 혈관 라벨 복원
    out_arr[ureter_mask] = label_array[ureter_mask]  # 원본 요관 라벨 복원

    # 분기 라벨 추가
    out_arr[renal_a.astype(bool)] = Label.RENAL_A
    out_arr[renal_v.astype(bool)] = Label.RENAL_V

    return copy_metadata(volume, out_arr)


def merge_tumor_to_kidney(volume: VolumeData) -> VolumeData:
    """
    Tumor 라벨을 Kidney 라벨로 병합

    메시 생성 시 신장 분할에 사용됩니다.

    Args:
        volume: 입력 VolumeData

    Returns:
        병합된 VolumeData
    """
    arr = volume.array.copy()
    arr[(arr == Label.TUMOR) | (arr == Label.KIDNEY)] = Label.KIDNEY
    return copy_metadata(volume, arr)
