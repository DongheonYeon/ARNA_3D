"""
NIfTI 파일 입출력 모듈

SimpleITK를 사용하여 NIfTI 파일을 읽고 씁니다.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk

from ..config.logger import logger
from ..config.constants import ResamplingParams
from ..domain.types import VolumeData


def load_nifti(file_path: Path | str) -> VolumeData | None:
    """
    NIfTI 파일을 VolumeData로 로드

    Args:
        file_path: NIfTI 파일 경로 (.nii 또는 .nii.gz)

    Returns:
        VolumeData 객체 (실패 시 None)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"VolumeLoadError: file not found: {file_path}")
        return None

    try:
        img = sitk.ReadImage(str(file_path))
        array = sitk.GetArrayFromImage(img)  # (Z, Y, X) 순서

        return VolumeData(
            array=array,
            spacing=img.GetSpacing(),      # (X, Y, Z) 순서
            origin=img.GetOrigin(),
            direction=img.GetDirection(),
        )
    except Exception as e:
        logger.error(f"VolumeLoadError: failed to load NIfTI: {file_path}", exception=e)
        return None


def save_nifti(volume: VolumeData, file_path: Path | str) -> Path:
    """
    VolumeData를 NIfTI 파일로 저장

    Args:
        volume: 저장할 VolumeData
        file_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    img = volume_to_sitk(volume)
    sitk.WriteImage(img, str(file_path))

    return file_path


def volume_to_sitk(volume: VolumeData) -> sitk.Image:
    """
    VolumeData를 SimpleITK Image로 변환

    Args:
        volume: 변환할 VolumeData

    Returns:
        SimpleITK Image 객체
    """
    img = sitk.GetImageFromArray(volume.array)
    img.SetSpacing(volume.spacing)
    img.SetOrigin(volume.origin)
    img.SetDirection(volume.direction)
    return img


def sitk_to_volume(img: sitk.Image) -> VolumeData:
    """
    SimpleITK Image를 VolumeData로 변환

    Args:
        img: SimpleITK Image 객체

    Returns:
        VolumeData 객체
    """
    return VolumeData(
        array=sitk.GetArrayFromImage(img),
        spacing=img.GetSpacing(),
        origin=img.GetOrigin(),
        direction=img.GetDirection(),
    )


def copy_metadata(source: VolumeData, target_array: np.ndarray) -> VolumeData:
    """
    source의 메타데이터를 유지하면서 새 array로 VolumeData 생성

    Args:
        source: 메타데이터를 가져올 VolumeData
        target_array: 새로운 array

    Returns:
        새 VolumeData 객체
    """
    return VolumeData(
        array=target_array.astype(source.array.dtype),
        spacing=source.spacing,
        origin=source.origin,
        direction=source.direction,
    )


def resample_volume(
    volume: VolumeData,
    target_spacing: tuple[float, float, float],
) -> VolumeData:
    """
    VolumeData를 목표 spacing으로 리샘플링

    세그멘테이션 마스크용으로 Nearest Neighbor 보간 사용

    Args:
        volume: 리샘플링할 VolumeData
        target_spacing: 목표 spacing (X, Y, Z) 순서

    Returns:
        리샘플링된 VolumeData
    """
    # VolumeData -> SimpleITK Image 변환
    original_img = volume_to_sitk(volume)

    # 원본 정보
    original_spacing = original_img.GetSpacing()
    original_size = original_img.GetSize()

    # 새 크기 계산 (물리적 크기 유지)
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    # 리샘플링 수행 (Nearest Neighbor - 세그멘테이션 마스크용)
    resampled_img = sitk.Resample(
        original_img,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        original_img.GetOrigin(),
        target_spacing,
        original_img.GetDirection(),
        0,  # default pixel value
        original_img.GetPixelID(),
    )

    return sitk_to_volume(resampled_img)


def resample_if_needed(
    volume: VolumeData,
    threshold: float | None = None,
    target_xy: float | None = None,
) -> VolumeData:
    """
    X, Y 축 중 하나라도 threshold 이하면 target_xy로 리샘플링

    Z축은 원본 유지

    Args:
        volume: 입력 VolumeData
        threshold: 리샘플링 기준 spacing (기본값: ResamplingParams.THRESHOLD)
        target_xy: 목표 X, Y spacing (기본값: ResamplingParams.TARGET_XY)

    Returns:
        리샘플링된 VolumeData (조건 미충족시 원본 반환)
    """
    # 기본값 설정
    params = ResamplingParams()
    threshold = threshold if threshold is not None else params.THRESHOLD
    target_xy = target_xy if target_xy is not None else params.TARGET_XY

    sx, sy, sz = volume.spacing

    # X 또는 Y 축 중 하나라도 threshold 이하인지 확인
    if sx > threshold and sy > threshold:
        logger.debug(f"Resampling not required: spacing(X,Y,Z)=({sx:.4f}, {sy:.4f}, {sz:.4f})mm")
        return volume

    nz, ny, nx = volume.shape
    logger.debug(f"Resampling required: spacing(X,Y,Z)=({sx:.4f}, {sy:.4f}, {sz:.4f})mm -> ({target_xy}, {target_xy}, {sz:.4f})mm")
    logger.debug(f"Original shape(X,Y,Z)=({nx}, {ny}, {nz})")

    # X, Y축만 target_xy로 변경, Z축은 유지
    target_spacing = (target_xy, target_xy, sz)
    resampled = resample_volume(volume, target_spacing)

    nz_new, ny_new, nx_new = resampled.shape
    logger.debug(f"Resampling complete: shape(X,Y,Z) ({nx}, {ny}, {nz}) -> ({nx_new}, {ny_new}, {nz_new})")

    return resampled
