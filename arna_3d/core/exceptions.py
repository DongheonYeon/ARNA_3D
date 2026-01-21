"""
커스텀 예외 정의

파이프라인에서 발생할 수 있는 예외들을 정의합니다.
"""


class Arna3DError(Exception):
    """ARNA-3D 기본 예외 클래스"""
    pass


class VolumeLoadError(Arna3DError):
    """NIfTI 볼륨 로딩 실패"""
    pass


class MeshExtractionError(Arna3DError):
    """메시 추출 실패"""
    pass


class VesselProcessingError(Arna3DError):
    """혈관 처리 실패"""
    pass


class SmoothingError(Arna3DError):
    """스무딩 처리 실패"""
    pass


class ConfigurationError(Arna3DError):
    """설정 관련 오류"""
    pass


class ValidationError(Arna3DError):
    """입력 데이터 검증 실패"""
    pass
