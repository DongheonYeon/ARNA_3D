"""
임시 파일 관리 모듈

Context manager를 사용하여 임시 파일을 안전하게 관리합니다.
"""

from pathlib import Path
from contextlib import contextmanager
import tempfile
import shutil
from typing import Generator
import SimpleITK as sitk

from ..core.types import VolumeData
from .nifti import volume_to_sitk


class TempFileManager:
    """
    임시 파일 관리자

    컨텍스트 매니저로 사용하여 임시 파일을 자동으로 정리합니다.
    """

    def __init__(self, prefix: str = "arna3d_", suffix: str = ".nii.gz"):
        """
        Args:
            prefix: 임시 파일 접두사
            suffix: 임시 파일 확장자
        """
        self.prefix = prefix
        self.suffix = suffix
        self._temp_files: list[Path] = []
        self._temp_dir: Path | None = None

    def __enter__(self) -> "TempFileManager":
        # 시스템 임시 디렉토리에 전용 폴더 생성
        self._temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """모든 임시 파일 및 디렉토리 삭제"""
        # 개별 파일 삭제
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"[WARN] 임시 파일 삭제 실패: {temp_file} - {e}")

        # 임시 디렉토리 삭제
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                print(f"[WARN] 임시 디렉토리 삭제 실패: {self._temp_dir} - {e}")

        self._temp_files.clear()
        self._temp_dir = None

    def create_temp_file(self, name: str | None = None) -> Path:
        """
        임시 파일 경로 생성 (파일은 생성하지 않음)

        Args:
            name: 파일 이름 (None이면 자동 생성)

        Returns:
            임시 파일 경로
        """
        if self._temp_dir is None:
            raise RuntimeError("TempFileManager를 컨텍스트 매니저로 사용하세요")

        if name is None:
            name = f"{self.prefix}{len(self._temp_files)}{self.suffix}"

        temp_path = self._temp_dir / name
        self._temp_files.append(temp_path)
        return temp_path

    def save_volume_temp(self, volume: VolumeData, name: str | None = None) -> Path:
        """
        VolumeData를 임시 NIfTI 파일로 저장

        Args:
            volume: 저장할 VolumeData
            name: 파일 이름 (None이면 자동 생성)

        Returns:
            임시 파일 경로
        """
        temp_path = self.create_temp_file(name)
        img = volume_to_sitk(volume)
        sitk.WriteImage(img, str(temp_path))
        return temp_path


@contextmanager
def temp_nifti_file(
    volume: VolumeData,
    prefix: str = "arna3d_",
) -> Generator[Path, None, None]:
    """
    VolumeData를 임시 NIfTI 파일로 저장하는 컨텍스트 매니저

    사용 예:
        with temp_nifti_file(volume) as temp_path:
            # temp_path로 작업
            ...
        # 블록 종료 시 자동 삭제

    Args:
        volume: 저장할 VolumeData
        prefix: 임시 파일 접두사

    Yields:
        임시 파일 경로
    """
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        temp_path = temp_dir / "temp.nii.gz"

        img = volume_to_sitk(volume)
        sitk.WriteImage(img, str(temp_path))

        yield temp_path

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
