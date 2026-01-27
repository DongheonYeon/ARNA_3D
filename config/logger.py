"""
백엔드 통합 시, 이 파일 전체를 다음과 같이 교체합니다:

from logger import logger

# Explicitly export logger for use by other modules in make_glb
__all__ = ['logger']

"""

class SimpleLogger:
    """
    백엔드 호환 경량 로거입니다.
    MedAILogger와 동일한 인터페이스를 제공합니다.
    """

    def __init__(self, name: str = "arna3d"):
        self.name = name

    def debug(self, message: str, **kwargs) -> None:
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log("WARN!", message, **kwargs)

    def error(self, message: str, exception: Exception = None, **kwargs) -> None:
        self._log("ERROR", message, **kwargs)
        if exception:
            print(f"         {type(exception).__name__}: {exception}")

    def critical(self, message: str, exception: Exception = None, **kwargs) -> None:
        self._log("CRITICAL", message, **kwargs)
        if exception:
            print(f"         {type(exception).__name__}: {exception}")

    def _log(self, level: str, message: str, **kwargs) -> None:
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        suffix = f" | {extra}" if extra else ""
        print(f"[{level:5}] {message}{suffix}")

logger = SimpleLogger()