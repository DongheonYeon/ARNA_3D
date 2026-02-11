from enum import IntEnum
from dataclasses import dataclass


class Label(IntEnum):
    """세그멘테이션 라벨 정의"""
    TUMOR = 1
    KIDNEY = 2
    ARTERY = 3
    VEIN = 4
    URETER = 5
    FAT = 6
    RENAL_A = 7  # 동맥 분기
    RENAL_V = 8  # 정맥 분기

    @classmethod
    def to_name_map(cls) -> dict[int, str]:
        """라벨 값 -> 이름 매핑"""
        return {label.value: label.name.capitalize() for label in cls}


@dataclass(frozen=True)
class VesselParams:
    """혈관 대동맥/신동맥 분할 파라미터"""
    # 동맥 파라미터
    ARTERY_PERCENTILE: int = 95
    ARTERY_THRESHOLD: float = 0.5  # 전체 슬라이스 대비 비율

    # 정맥 파라미터
    VEIN_PERCENTILE: int = 90
    VEIN_THRESHOLD: float = 0.7  # 전체 슬라이스 대비 비율

    # 보간 파라미터
    DILATION_ITERATIONS: int = 3  # 원형/타원 보간 시 dilation 반복 횟수

    # Z-score 분석 파라미터
    ZSCORE_WINDOW_SIZE: int = 5

    # 신동맥/신정맥 부피 필터링 (mm³)
    RENAL_MIN_VOLUME_MM3: float = 100.0


@dataclass(frozen=True)
class PoissonParams:
    """Poisson Reconstruction 파라미터"""
    DEPTH: int = 8
    SAMPLE_POINTS: int = 40000


@dataclass(frozen=True)
class MorphologyParams:
    """Morphology 연산 파라미터"""
    # Fat dilation
    FAT_DILATION_ITERATIONS: int = 2
    
    # Tumor erosion (kidney 보호 거리 기반)
    TUMOR_EROSION_ITERATIONS: int = 2
    TUMOR_PROTECT_DISTANCE: int = 5
    KIDNEY_COMPONENT_MIN_SIZE: int = 10  # kidney 작은 조각 제거 임계값 (voxels)

    # 메시 분할
    MAX_BILATERAL_PARTS: int = 2  # L/R 분할 시 최대 파트 수


@dataclass(frozen=True)
class TumorParams:
    """종양 필터링 파라미터"""
    # 최소 부피 (mm³) - 이보다 작은 종양은 제거
    MIN_VOLUME_MM3: float = 0


@dataclass(frozen=True)
class SmoothingFuncType:
    """Smoothing 함수 타입 상수"""
    LAPLACIAN: str = "laplacian"
    TAUBIN: str = "taubin"


@dataclass(frozen=True)
class DilationFuncType:
    """Dilation 함수 타입 상수"""
    DEFAULT: str = "default"


@dataclass(frozen=True)
class SimplificationFuncType:
    """Simplification 함수 타입 상수"""
    DECIMATE: str = "decimate"
    DECIMATE_PRO: str = "decimate_pro"


@dataclass(frozen=True)
class ResamplingParams:
    """고해상도 이미지 리샘플링 파라미터"""
    # X, Y 축 중 하나라도 이 값 이하면 리샘플링 수행
    THRESHOLD: float = 0.25  # mm
    # 리샘플링 목표 spacing (X, Y축에만 적용, Z축은 원본 유지)
    TARGET_XY: float = 0.75  # mm
