import time
from pathlib import Path
from dataclasses import dataclass, field

from .config.settings import PipelineSettings, SmoothingPreset
from .config.logger import logger
from .domain.types import VolumeData, MeshCollection
from .file_io.nifti import load_nifti, resample_if_needed
from .file_io.mesh import save_scene, save_debug_scene
from .threeDrecon.segmentation.preprocessing import (
    preprocess_segmentation,
    preprocess_kidney_segmentation,
)
from .threeDrecon.mesh.extraction import extract_meshes_from_volume
from .threeDrecon.mesh.smoothing import smooth_mesh_collection
from .threeDrecon.mesh.reconstruction import process_vessel_reconstruction


@dataclass
class Pipeline:
    """
    ARNA-3D 처리 파이프라인

    NIfTI 세그멘테이션을 3D GLB 모델로 변환하는 전체 프로세스를 관리합니다.
    """
    settings: PipelineSettings

    # 내부 상태
    _volume: VolumeData | None = field(default=None, init=False)
    _meshes: MeshCollection | None = field(default=None, init=False)

    def run(self) -> str | None:
        """
        파이프라인 실행

        Returns:
            저장된 GLB 파일 경로 (실패 시 None)
        """
        start_time = time.time()

        logger.debug(f"Starting 3D reconstruction process: {self.settings.input_path}")
        logger.debug(f"Case ID: {self.settings.case_id}, Phase: {self.settings.phase}")

        # 1. NIfTI 로드
        self._volume = load_nifti(self.settings.input_path)
        if self._volume is None:
            logger.error("VolumeLoadError: pipeline aborted due to NIfTI load failure")
            return None

        # 2. 고해상도 이미지 리샘플링 (X, Y 중 하나라도 0.25mm 이하면 0.75mm로)
        logger.debug("Checking resolution...")
        self._volume = resample_if_needed(self._volume)

        # 3. 세그멘테이션 전처리
        logger.debug("Preprocessing segmentation file...")
        processed_volume = preprocess_segmentation(self._volume)
        kidney_volume = preprocess_kidney_segmentation(processed_volume)

        # 4. 메시 추출 (in-memory)
        logger.debug("Extracting Mesh from volume...")
        self._meshes = extract_meshes_from_volume(processed_volume, kidney_volume)

        # 디버그 저장 (step1 전)
        self._save_debug("before_step1")

        # 5. 1단계 스무딩
        logger.debug("Step 1 in progress...")
        stage1_preset = self.settings.load_stage1_preset()
        self._meshes = smooth_mesh_collection(self._meshes, list(stage1_preset))

        self._save_debug("after_step1")

        # 6. Poisson 재구성
        logger.debug("Applying poisson reconstruction...")
        self._meshes = process_vessel_reconstruction(self._meshes)

        # 7. 2단계 스무딩
        logger.debug("Step 2 in progress...")
        stage2_preset = self.settings.load_stage2_preset()
        self._meshes = smooth_mesh_collection(self._meshes, list(stage2_preset))

        # 8. 결과 저장
        output_path = self._save_result()

        elapsed = time.time() - start_time
        logger.debug(f"Process complete. Elapsed time: {elapsed:.2f}s")
        logger.debug(f"Output path: {output_path}")

        return str(output_path)

    def _save_debug(self, tag: str) -> None:
        """디버그용 중간 결과 저장"""
        if not self.settings.debug or self._meshes is None:
            return

        scene = self._meshes.to_scene()
        save_debug_scene(
            scene,
            self.settings.output_path.parent,
            self.settings.phase or "X",
            tag,
            debug=True,
        )

    def _save_result(self) -> Path:
        """최종 결과 저장"""
        output_path = self.settings.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = self._meshes.to_scene()
        save_scene(scene, output_path)

        return output_path


def run_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    debug: bool = False,
) -> str | None:
    """
    파이프라인 실행 함수

    Args:
        input_path: 입력 NIfTI 파일 경로
        output_path: 출력 GLB 파일 경로
        debug: 디버그 모드

    Returns:
        저장된 GLB 파일 경로 (실패 시 None)
    """
    settings = PipelineSettings(
        input_path=Path(input_path),
        output_path=Path(output_path),
        debug=debug,
    )

    pipeline = Pipeline(settings=settings)
    return pipeline.run()
