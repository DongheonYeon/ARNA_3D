"""
혈관 보간 모듈

분기점 사이를 원형 또는 타원 보간으로 연결합니다.
"""

import numpy as np
import cv2
from scipy.ndimage import binary_dilation

from config.constants import VesselParams
from threeDrecon.vessel.analysis import get_largest_component, get_max_inscribed_circle


def get_fitted_ellipse(mask_2d: np.ndarray) -> tuple | None:
    """
    2D 마스크에 타원 피팅

    Args:
        mask_2d: 2D 바이너리 마스크

    Returns:
        ((cx, cy), (major, minor), angle) 또는 None
    """
    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    return cv2.fitEllipse(cnt)


def draw_ellipse_mask(shape: tuple[int, int], ellipse: tuple | None) -> np.ndarray:
    """
    타원 마스크 그리기

    Args:
        shape: 출력 마스크 shape (H, W)
        ellipse: 타원 파라미터 ((cx, cy), (major, minor), angle)

    Returns:
        타원이 채워진 바이너리 마스크
    """
    canvas = np.zeros(shape, dtype=np.uint8)

    if ellipse is None:
        return canvas

    pts = cv2.ellipse2Poly(
        center=(int(ellipse[0][0]), int(ellipse[0][1])),
        axes=(int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
        angle=int(ellipse[2]),
        arcStart=0,
        arcEnd=360,
        delta=1,
    )
    cv2.fillConvexPoly(canvas, pts, 1)

    return canvas


def interpolate_circle_bridge(
    mask_3d: np.ndarray,
    z_front: int,
    z_back: int,
    dilation_iterations: int = VesselParams.DILATION_ITERATIONS,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    원형 보간으로 분기점 사이를 연결

    동맥처럼 단면이 원형에 가까운 혈관에 적합합니다.

    Args:
        mask_3d: 3D 바이너리 마스크
        z_front: 분기 시작 Z
        z_back: 분기 끝 Z
        dilation_iterations: dilation 반복 횟수

    Returns:
        (보간된 마스크, (실제 시작 Z, 실제 끝 Z))
    """
    z0, z1 = z_front - 1, z_back + 1
    n = z1 - z0 + 1
    interp = np.zeros((n, *mask_3d.shape[1:]), dtype=np.uint8)

    # 시작/끝 슬라이스의 내접원 정보
    c0, r0 = get_max_inscribed_circle(mask_3d[z0])
    c1, r1 = get_max_inscribed_circle(mask_3d[z1])

    for i in range(n):
        if c0 is None or c1 is None:
            continue

        alpha = i / (n - 1) if n > 1 else 0

        # 선형 보간
        y = int((1 - alpha) * c0[0] + alpha * c1[0])
        x = int((1 - alpha) * c0[1] + alpha * c1[1])
        r = (1 - alpha) * r0 + alpha * r1

        # 원형 마스크 생성
        Y, X = np.ogrid[: mask_3d.shape[1], : mask_3d.shape[2]]
        circle = ((Y - y) ** 2 + (X - x) ** 2) <= r ** 2

        # dilation 적용
        dil = binary_dilation(circle, iterations=dilation_iterations)
        interp[i] = dil.astype(np.uint8)

    # 결과 합치기
    out = mask_3d.copy()
    out[z0 : z1 + 1] = interp
    out = get_largest_component(out, n=1)

    return out, (z0, z1)


def interpolate_ellipse_bridge(
    mask_3d: np.ndarray,
    z_front: int,
    z_back: int,
    dilation_iterations: int = VesselParams.DILATION_ITERATIONS,
) -> np.ndarray:
    """
    타원 보간으로 분기점 사이를 연결

    정맥처럼 단면이 타원형에 가까운 혈관에 적합합니다.

    Args:
        mask_3d: 3D 바이너리 마스크
        z_front: 분기 시작 Z
        z_back: 분기 끝 Z
        dilation_iterations: dilation 반복 횟수

    Returns:
        보간된 마스크
    """
    z0, z1 = z_front - 1, z_back + 1
    n_slices = z1 - z0 + 1

    orig = mask_3d.copy().astype(np.uint8)
    e0 = get_fitted_ellipse(orig[z0])
    e1 = get_fitted_ellipse(orig[z1])

    interp_stack = np.zeros((n_slices, *mask_3d.shape[1:]), dtype=np.uint8)

    for i in range(n_slices):
        if e0 is None or e1 is None:
            continue

        alpha = i / (n_slices - 1) if n_slices > 1 else 0

        # 선형 보간: 중심, 축, 각도
        cx = (1 - alpha) * e0[0][0] + alpha * e1[0][0]
        cy = (1 - alpha) * e0[0][1] + alpha * e1[0][1]
        major = (1 - alpha) * e0[1][0] + alpha * e1[1][0]
        minor = (1 - alpha) * e0[1][1] + alpha * e1[1][1]
        angle = (1 - alpha) * e0[2] + alpha * e1[2]

        ellipse = ((cx, cy), (major, minor), angle)
        mask2d = draw_ellipse_mask(orig.shape[1:], ellipse)
        dil = binary_dilation(mask2d, iterations=dilation_iterations).astype(np.uint8)
        interp_stack[i] = dil

    # 결과 합치기
    bridged = orig.copy()
    bridged[z0 : z1 + 1] = interp_stack
    bridged = get_largest_component(bridged, n=1)

    return bridged
