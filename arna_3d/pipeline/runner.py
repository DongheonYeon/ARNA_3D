"""
파이프라인 실행 모듈

전체 처리 파이프라인을 오케스트레이션합니다.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field

from ..config.settings import PipelineSettings, SmoothingPreset
from ..core.types import VolumeData, MeshCollection
from ..io.nifti import load_nifti
from ..io.mesh import save_scene, save_debug_scene
from ..io.temp import TempFileManager
from ..processing.segmentation import preprocess_segmentation, merge_tumor_to_kidney
from ..processing.mesh.extraction import extract_meshes_from_volume
from ..processing.mesh.smoothing import smooth_mesh_collection
from ..processing.mesh.reconstruction import process_vessel_reconstruction


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

    def run(self) -> Path:
        """
        파이프라인 실행

        Returns:
            저장된 GLB 파일 경로
        """
        start_time = time.time()

        print(f"[INFO] 입력 파일: {self.settings.input_path}")
        print(f"[INFO] Case ID: {self.settings.case_id}, Phase: {self.settings.phase}")

        # 1. NIfTI 로드
        self._volume = load_nifti(self.settings.input_path)

        # 2. 세그멘테이션 전처리
        print("[INFO] 세그멘테이션 전처리 중...")
        processed_volume = preprocess_segmentation(self._volume)

        # 3. 신장용 볼륨 생성 (Tumor → Kidney 병합)
        kidney_volume = merge_tumor_to_kidney(processed_volume)

        # 4. 임시 파일로 저장 후 메시 추출
        with TempFileManager(prefix="arna3d_") as temp_mgr:
            temp_nifti = temp_mgr.save_volume_temp(processed_volume, "processed.nii.gz")
            temp_kidney = temp_mgr.save_volume_temp(kidney_volume, "kidney.nii.gz")

            print("[INFO] 메시 추출 중...")
            self._meshes = extract_meshes_from_volume(temp_nifti, temp_kidney)

        # 디버그 저장 (step1 전)
        self._save_debug("before_step1")

        # 5. 1단계 스무딩
        print("[INFO] Step 1: 스무딩 적용 중...")
        stage1_preset = self.settings.load_stage1_preset()
        self._meshes = smooth_mesh_collection(self._meshes, list(stage1_preset))

        self._save_debug("after_step1")

        # 6. Poisson 재구성
        print("[INFO] Poisson 재구성 중...")
        self._meshes = process_vessel_reconstruction(self._meshes)

        # 7. 2단계 스무딩
        print("[INFO] Step 2: 마무리 스무딩 중...")
        stage2_preset = self.settings.load_stage2_preset()
        self._meshes = smooth_mesh_collection(self._meshes, list(stage2_preset))

        # 8. 결과 저장
        output_path = self._save_result()

        elapsed = time.time() - start_time
        print(f"[INFO] 완료. 소요 시간: {elapsed:.2f}초")
        print(f"[INFO] 저장 위치: {output_path}")

        return output_path

    def _save_debug(self, tag: str) -> None:
        """디버그용 중간 결과 저장"""
        if not self.settings.debug or self._meshes is None:
            return

        scene = self._meshes.to_scene()
        save_debug_scene(
            scene,
            self.settings.output_dir,
            self.settings.phase or "X",
            tag,
            debug=True,
        )

    def _save_result(self) -> Path:
        """최종 결과 저장"""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        phase = self.settings.phase or "X"
        output_path = self.settings.output_dir / f"obj_{phase}.glb"

        scene = self._meshes.to_scene()
        save_scene(scene, output_path)

        return output_path


def run_pipeline(
    input_path: Path | str,
    output_dir: Path | str | None = None,
    debug: bool = False,
) -> Path:
    """
    파이프라인 실행 편의 함수

    Args:
        input_path: 입력 NIfTI 파일 경로
        output_dir: 출력 디렉토리 (None이면 자동 설정)
        debug: 디버그 모드

    Returns:
        저장된 GLB 파일 경로
    """
    settings = PipelineSettings(
        input_path=Path(input_path),
        output_dir=Path(output_dir) if output_dir else None,
        debug=debug,
    )

    pipeline = Pipeline(settings=settings)
    return pipeline.run()
