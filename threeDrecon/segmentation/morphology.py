"""
형태학적 연산 모듈

공통으로 사용되는 형태학적 연산을 제공합니다.
"""

import numpy as np
import scipy.ndimage
from skimage.measure import regionprops

from config.constants import MorphologyParams


def get_largest_component(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """
    가장 큰 연결 요소 n개 추출

    Args:
        mask: 바이너리 마스크 (2D 또는 3D)
        n: 추출할 연결 요소 개수

    Returns:
        가장 큰 n개의 연결 요소만 포함하는 마스크
    """
    if mask.ndim == 2:
        structure = np.ones((3, 3), dtype=bool)
    elif mask.ndim == 3:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError("입력 마스크는 2D 또는 3D여야 합니다")

    labeled, _ = scipy.ndimage.label(mask, structure=structure)
    props = regionprops(labeled)

    if not props or n <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)
    labels = [p.label for p in props_sorted[:n]]

    result = np.zeros_like(mask, dtype=np.uint8)
    for lbl in labels:
        result[labeled == lbl] = 1

    return result
