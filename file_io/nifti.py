"""
NIfTI 파일 입출력 모듈

SimpleITK를 사용하여 NIfTI 파일을 읽고 씁니다.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk

from core.types import VolumeData
from core.exceptions import VolumeLoadError


def load_nifti(file_path: Path | str) -> VolumeData:
    """
    NIfTI 파일을 VolumeData로 로드

    Args:
        file_path: NIfTI 파일 경로 (.nii 또는 .nii.gz)

    Returns:
        VolumeData 객체

    Raises:
        VolumeLoadError: 파일 로드 실패 시
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise VolumeLoadError(f"파일을 찾을 수 없습니다: {file_path}")

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
        raise VolumeLoadError(f"NIfTI 로드 실패: {file_path} - {e}") from e


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
