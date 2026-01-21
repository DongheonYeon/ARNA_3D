"""세그멘테이션 처리 모듈"""

from .preprocessing import preprocess_segmentation, apply_fat_dilation
from .morphology import get_largest_component

__all__ = [
    "preprocess_segmentation",
    "apply_fat_dilation",
    "get_largest_component",
]
