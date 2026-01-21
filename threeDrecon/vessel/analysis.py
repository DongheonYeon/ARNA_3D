"""
혈관 반경 분석 모듈

혈관 마스크에서 반경 배열을 계산하고 분기점 범위를 탐지합니다.
"""

import numpy as np
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from scipy.stats import zscore
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

from config.constants import VesselParams


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


def get_max_inscribed_circle(mask_2d: np.ndarray) -> tuple[tuple[int, int] | None, float | None]:
    """
    2D 마스크에서 최대 내접원의 중심과 반경 계산

    Args:
        mask_2d: 2D 바이너리 마스크

    Returns:
        (중심 좌표 (y, x), 반경) 또는 (None, None)
    """
    labeled, _ = scipy.ndimage.label(mask_2d)
    if labeled.max() == 0:
        return None, None

    # 가장 큰 연결 요소 선택
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lbl = sizes.argmax()
    cc = labeled == lbl

    # Distance transform으로 최대 내접원 찾기
    dist = distance_transform_edt(cc)
    y, x = np.unravel_index(dist.argmax(), dist.shape)

    return (int(y), int(x)), float(dist.max())


def compute_radii_array(
    mask_3d: np.ndarray,
    z_start: int,
    z_end: int,
) -> np.ndarray:
    """
    Z축을 따라 각 슬라이스의 내접원/외접원 반경 배열 계산

    Args:
        mask_3d: 3D 바이너리 마스크
        z_start: 시작 Z 인덱스
        z_end: 끝 Z 인덱스

    Returns:
        shape (z_dim, 2)의 배열, [:, 0]은 내접원 반경, [:, 1]은 외접원 반경
    """
    z_dim = mask_3d.shape[0]
    radii = np.zeros((z_dim, 2))

    for z in range(z_start, z_end + 1):
        slice_mask = get_largest_component(mask_3d[z])
        if slice_mask.sum() == 0:
            continue

        # 최대 내접원 반경
        _, r_max = get_max_inscribed_circle(slice_mask)

        # 최소 외접원 반경 (convex hull centroid에서 가장 먼 점)
        ch = convex_hull_image(slice_mask)
        props = regionprops(label(ch))
        if props:
            cy, cx = props[0].centroid
            coords = props[0].coords
            dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
            r_min = dists.max()
        else:
            r_min = 0

        radii[z] = [r_max if r_max else 0, r_min]

    return radii


def detect_gradient_range(
    mask_3d: np.ndarray,
    z_start: int,
    z_end: int,
    percentile: int = VesselParams.ARTERY_PERCENTILE,
) -> tuple[int, int]:
    """
    그래디언트 기반 분기점 범위 탐지

    반경 차이의 그래디언트가 급격히 변하는 지점을 찾습니다.

    Args:
        mask_3d: 3D 바이너리 마스크
        z_start: 분석 시작 Z 인덱스
        z_end: 분석 끝 Z 인덱스
        percentile: 임계값 계산용 백분위수

    Returns:
        (분기 시작 Z, 분기 끝 Z)
    """
    radii = compute_radii_array(mask_3d, z_start, z_end)

    # 유효한 슬라이스만 사용
    valid = np.all(radii > 0, axis=1)
    grad = np.zeros_like(valid, dtype=float)

    if valid.sum() >= 2:
        grad_sub = np.gradient(radii[valid, 1] - radii[valid, 0])
        grad[valid] = np.abs(grad_sub)

    # 백분위수 기반 임계값 계산
    valid_grads = grad[valid]
    threshold = np.percentile(valid_grads, percentile) if len(valid_grads) > 0 else 0

    # 임계값 이상인 첫/마지막 위치 찾기
    above_threshold = grad >= threshold
    z_front = np.argmax(above_threshold)
    z_back = len(grad) - 1 - np.argmax(above_threshold[::-1])

    return int(z_front), int(z_back)


def detect_zscore_range(
    mask_3d: np.ndarray,
    z_start: int,
    z_end: int,
    window_size: int = VesselParams.ZSCORE_WINDOW_SIZE,
) -> tuple[int | None, int | None]:
    """
    Z-score 기반 분기점 범위 탐지

    반경 차이의 Z-score가 연속으로 0 이상인 구간을 찾습니다.

    Args:
        mask_3d: 3D 바이너리 마스크
        z_start: 분석 시작 Z 인덱스
        z_end: 분석 끝 Z 인덱스
        window_size: 연속 구간 탐지 윈도우 크기

    Returns:
        (분기 시작 Z, 분기 끝 Z) 또는 (None, None)
    """
    radii = compute_radii_array(mask_3d, z_start, z_end)
    z = np.arange(radii.shape[0])
    max_r = radii[:, 0]
    min_r = radii[:, 1]

    # 유효 영역 필터링
    valid_mask = (min_r > 0) & (max_r > 0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < window_size:
        return None, None

    radius_diff_valid = np.abs(min_r[valid_mask] - max_r[valid_mask])
    z_scores_diff_valid = zscore(radius_diff_valid)

    # 연속된 0 이상 구간의 시작점 찾기
    z_front = None
    for i in range(len(z_scores_diff_valid) - window_size + 1):
        window = z_scores_diff_valid[i : i + window_size]
        if np.all(window >= 0):
            z_front = z[valid_indices[i]]
            break

    if z_front is None:
        fallback_idx = np.where(z_scores_diff_valid >= 0)[0]
        z_front = z[valid_indices[fallback_idx[0]]] if len(fallback_idx) > 0 else None

    # 연속된 0 이상 구간의 끝점 찾기
    z_back = None
    for i in range(len(z_scores_diff_valid) - window_size, -1, -1):
        window = z_scores_diff_valid[i : i + window_size]
        if np.all(window >= 0):
            z_back = z[valid_indices[i + window_size - 1]]
            break

    if z_back is None:
        fallback_idx = np.where(z_scores_diff_valid >= 0)[0]
        z_back = z[valid_indices[fallback_idx[-1]]] if len(fallback_idx) > 0 else None

    return z_front, z_back
