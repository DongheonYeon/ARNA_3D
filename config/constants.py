"""
상수 정의 모듈

모든 매직 넘버와 라벨 정의를 중앙에서 관리합니다.
"""

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
        """라벨 값 → 이름 매핑"""
        return {label.value: label.name.capitalize() for label in cls}

    @classmethod
    def to_value_map(cls) -> dict[str, int]:
        """이름 → 라벨 값 매핑 (기존 LABELS 딕셔너리 호환)"""
        return {
            "Tumor": cls.TUMOR,
            "Kidney": cls.KIDNEY,
            "Artery": cls.ARTERY,
            "Vein": cls.VEIN,
            "Ureter": cls.URETER,
            "Fat": cls.FAT,
            "Renal_a": cls.RENAL_A,
            "Renal_v": cls.RENAL_V,
        }


# 기존 코드 호환용 LABELS 딕셔너리
LABELS = Label.to_value_map()


@dataclass(frozen=True)
class VesselParams:
    """혈관 분석 파라미터"""
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


@dataclass(frozen=True)
class PoissonParams:
    """Poisson 재구성 파라미터"""
    DEPTH: int = 8
    SAMPLE_POINTS: int = 40000


@dataclass(frozen=True)
class MorphologyParams:
    """형태학적 연산 파라미터"""
    # Fat dilation
    FAT_DILATION_ITERATIONS: int = 2

    # 연결 요소 분석
    CONNECTIVITY_2D: int = 8   # 2D 연결성 (8-neighbor)
    CONNECTIVITY_3D: int = 26  # 3D 연결성 (26-neighbor)

    # 메시 분할
    MAX_BILATERAL_PARTS: int = 2  # L/R 분할 시 최대 파트 수


@dataclass(frozen=True)
class SmoothingFuncType:
    """스무딩 함수 타입 상수"""
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
