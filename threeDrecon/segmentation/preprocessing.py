"""
세그멘테이션 전처리 모듈

NIfTI 세그멘테이션 라벨을 전처리합니다.
"""

from pathlib import Path
import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_dilation, distance_transform_edt

from ...config.constants import Label, MorphologyParams, VesselParams, TumorParams
from ...config.logger import logger
from ...domain.types import VolumeData
from ...file_io.nifti import copy_metadata, save_nifti
from ..vessel.branch import process_vessel_branches


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


def filter_by_volume(
    mask: np.ndarray,
    min_volume_mm3: float,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """
    최소 부피 이하의 연결 요소 제거

    Args:
        mask: 바이너리 마스크 (3D)
        min_volume_mm3: 최소 부피 임계값 (mm³)
        spacing: 복셀 간격 (z, y, x) in mm

    Returns:
        필터링된 마스크
    """
    if not mask.any():
        return mask

    voxel_volume = float(np.prod(spacing))  # mm³ per voxel
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_components = scipy.ndimage.label(mask, structure=structure)

    if n_components == 0:
        return mask

    result = np.zeros_like(mask, dtype=np.uint8)
    removed_count = 0

    for i in range(1, n_components + 1):
        component = labeled == i
        volume_mm3 = component.sum() * voxel_volume
        if volume_mm3 >= min_volume_mm3:
            result[component] = 1
        else:
            removed_count += 1

    if removed_count > 0:
        logger.debug(f"Volume filter: removed {removed_count} components < {min_volume_mm3:.1f}mm³")

    return result


def filter_tumor(
    label_array: np.ndarray,
    spacing: tuple[float, float, float],
    min_volume_mm3: float = TumorParams.MIN_VOLUME_MM3,
) -> np.ndarray:
    """
    종양 필터링: 신장과 접촉하지 않거나 너무 작은 종양 제거

    Args:
        label_array: 세그멘테이션 라벨 배열
        spacing: 복셀 간격 (X, Y, Z) in mm
        min_volume_mm3: 최소 부피 임계값 (mm³)

    Returns:
        필터링된 라벨 배열
    """
    tumor_mask = label_array == Label.TUMOR
    kidney_mask = label_array == Label.KIDNEY

    if not tumor_mask.any():
        return label_array

    if not kidney_mask.any():
        # 신장이 없으면 모든 종양 제거
        logger.debug("Tumor filter: no kidney found, removing all tumors")
        result = label_array.copy()
        result[tumor_mask] = 0
        return result

    # 신장 마스크를 1 voxel dilation (접촉 판정용)
    structure = np.ones((3, 3, 3), dtype=bool)
    kidney_dilated = binary_dilation(kidney_mask, structure=structure, iterations=1)

    # 개별 종양 컴포넌트 분리
    labeled, n_tumors = scipy.ndimage.label(tumor_mask, structure=structure)

    if n_tumors == 0:
        return label_array

    voxel_volume = float(np.prod(spacing))  # mm³ per voxel
    result = label_array.copy()

    removed_no_contact = 0
    removed_volume = 0

    for i in range(1, n_tumors + 1):
        component = labeled == i

        # 1. 신장과 접촉 여부 체크 (dilated 신장과 교집합)
        if not np.any(component & kidney_dilated):
            result[component] = 0
            removed_no_contact += 1
            continue

        # 2. 부피 체크 (min_volume_mm3 > 0 일 때만)
        if min_volume_mm3 > 0:
            volume_mm3 = component.sum() * voxel_volume
            if volume_mm3 < min_volume_mm3:
                result[component] = 0
                removed_volume += 1
                continue

    if removed_no_contact > 0 or removed_volume > 0:
        logger.debug(f"Tumor filter: removed {removed_no_contact} (no kidney contact) + {removed_volume} (< {min_volume_mm3:.1f}mm³)")

    return result


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


def preprocess_segmentation(
    volume: VolumeData,
    min_renal_volume_mm3: float = VesselParams.RENAL_MIN_VOLUME_MM3,
    enable_vessel_branch_split: bool = True,
) -> VolumeData:
    """
    세그멘테이션 전처리 메인 함수

    1. 종양 필터링 (신장 bbox 외부, 너무 작은 종양 제거)
    2. 혈관 분기 자동 분할 (Renal_a, Renal_v 생성)
    3. 분기에 부피 필터링 적용
    4. Fat dilation 적용

    Args:
        volume: 입력 VolumeData
        min_renal_volume_mm3: 신동맥/신정맥 최소 부피 임계값 (mm³)
        enable_vessel_branch_split: True면 혈관 분기(split) 수행

    Returns:
        전처리된 VolumeData
    """
    # 종양 필터링 (신장 bbox 외부 + 작은 부피 제거)
    label_array = filter_tumor(volume.array, volume.spacing)

    renal_a = np.zeros_like(label_array, dtype=np.uint8)
    renal_v = np.zeros_like(label_array, dtype=np.uint8)

    if enable_vessel_branch_split:
        # 신장 Z 범위 계산
        z_start, z_end = get_kidney_z_range(label_array)

        # 혈관 분기 처리
        renal_a, renal_v = process_vessel_branches(label_array, z_start, z_end)

        # 부피 필터링 적용 (작은 노이즈 제거)
        if min_renal_volume_mm3 > 0:
            renal_a = filter_by_volume(renal_a, min_renal_volume_mm3, volume.spacing)
            renal_v = filter_by_volume(renal_v, min_renal_volume_mm3, volume.spacing)
    else:
        logger.info("Vessel branch split disabled: skip Renal_a/Renal_v generation.")

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
    if enable_vessel_branch_split:
        out_arr[renal_a.astype(bool)] = Label.RENAL_A
        out_arr[renal_v.astype(bool)] = Label.RENAL_V

    return copy_metadata(volume, out_arr)


def preprocess_kidney_segmentation(
    volume: VolumeData,
    protect_distance: int = MorphologyParams.TUMOR_PROTECT_DISTANCE
) -> VolumeData:
    """
    Kidney 경계에서 protect_distance 이내의 Tumor 영역을 Kidney 라벨로 병합

    Args:
        volume: 입력 VolumeData
        protect_distance: kidney로부터 보호할 거리(voxel)

    Returns:
        병합된 VolumeData
    """
    arr = volume.array.copy()
    tumor_mask = arr == Label.TUMOR
    kidney_mask = arr == Label.KIDNEY

    if protect_distance > 0:
        distance_to_kidney = distance_transform_edt(~kidney_mask)
        allow_erosion = distance_to_kidney > protect_distance
    else:
        allow_erosion = np.ones_like(kidney_mask, dtype=bool)

    tumor_merge = tumor_mask & ~allow_erosion
    
    arr[tumor_mask] = 0
    arr[kidney_mask | tumor_merge] = Label.KIDNEY
    return copy_metadata(volume, arr)
"""
def preprocess_kidney_segmentation_with_erosion(
    volume: VolumeData,
    erosion_iterations: int = MorphologyParams.TUMOR_EROSION_ITERATIONS,
    protect_distance: int = MorphologyParams.TUMOR_PROTECT_DISTANCE,
    debug: bool = False,
    debug_dir: Path | str | None = None,
    phase: str | None = None,
) -> VolumeData:

    # Tumor 라벨을 거리 기반 보호 후 erosion하여 Kidney 라벨로 병합

    # Kidney 경계에서 protect_distance 이내의 Tumor 영역은 erosion에서 보호합니다.

    # Args:
    #     volume: 입력 VolumeData
    #     erosion_iterations: tumor erosion 반복 횟수
    #     protect_distance: kidney로부터 보호할 거리(voxel)
    #     debug: True면 디버그용 NIfTI 저장
    #     debug_dir: 디버그 저장 디렉토리
    #     phase: 디버그 파일명에 사용할 phase

    # Returns:
    #     병합된 VolumeData

    arr = volume.array.copy()
    tumor_mask = arr == Label.TUMOR
    kidney_mask = arr == Label.KIDNEY

    if protect_distance > 0:
        distance_to_kidney = distance_transform_edt(~kidney_mask)
        allow_erosion = distance_to_kidney > protect_distance
    else:
        allow_erosion = np.ones_like(kidney_mask, dtype=bool)

    structure = np.ones((3, 3, 3), dtype=bool)
    eroded_tumor = binary_erosion(
        tumor_mask,
        structure=structure,
        iterations=erosion_iterations,
    )

    # 보호 영역은 그대로 유지
    protected_tumor = tumor_mask & ~allow_erosion
    tumor_final = (eroded_tumor & allow_erosion) | protected_tumor

    if debug and debug_dir is not None:
        debug_arr = np.zeros_like(volume.array)
        debug_arr[kidney_mask | tumor_final] = Label.KIDNEY
        debug_volume = copy_metadata(volume, debug_arr)
        phase_tag = phase or "X"
        debug_path = Path(debug_dir) / f"seg_{phase_tag}_kidney_erosion.nii.gz"
        try:
            save_nifti(debug_volume, debug_path)
            logger.debug(f"Saved: {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save kidney erosion debug NIfTI: {e}")

    # tumor 라벨 제거 후 kidney와 병합
    arr[tumor_mask] = 0
    arr[kidney_mask | tumor_final] = Label.KIDNEY
    return copy_metadata(volume, arr)
"""
