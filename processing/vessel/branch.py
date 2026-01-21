"""
혈관 분기 추출 모듈

혈관에서 분기(branch)를 추출합니다.
"""

import numpy as np

from config.constants import Label, VesselParams
from core.exceptions import VesselProcessingError
from processing.vessel.analysis import get_largest_component, detect_gradient_range
from processing.vessel.interpolation import interpolate_circle_bridge, interpolate_ellipse_bridge


def extract_branches(
    original: np.ndarray,
    bridged: np.ndarray,
    top_n: int = 2,
) -> np.ndarray:
    """
    원본과 보간된 마스크의 차이에서 분기 추출

    Args:
        original: 원본 혈관 마스크
        bridged: 보간된 혈관 마스크
        top_n: 추출할 최대 분기 개수

    Returns:
        분기 마스크
    """
    llc = get_largest_component(bridged, n=1)
    branches = (original.astype(bool) & ~llc.astype(bool)).astype(np.uint8)
    return branches


def process_artery_branches(
    artery_mask: np.ndarray,
    z_start: int,
    z_end: int,
    percentile: int = VesselParams.ARTERY_PERCENTILE,
    threshold: float = VesselParams.ARTERY_THRESHOLD,
) -> np.ndarray:
    """
    동맥 분기 처리

    Args:
        artery_mask: 동맥 바이너리 마스크
        z_start: 분석 시작 Z (신장 범위)
        z_end: 분석 끝 Z (신장 범위)
        percentile: 그래디언트 임계값 백분위수
        threshold: 전체 슬라이스 대비 허용 비율

    Returns:
        신동맥(Renal_a) 분기 마스크
    """
    if not artery_mask.any():
        return np.zeros_like(artery_mask)

    total_slices = artery_mask.shape[0]
    zf, zb = detect_gradient_range(artery_mask, z_start, z_end, percentile=percentile)

    artery_range = zb - zf + 1 if (zf is not None and zb is not None) else total_slices
    print(f"[INFO] Artery: index=[{zf}-{zb}], gradient range={artery_range}/{total_slices} ({artery_range/total_slices*100:.1f}%)")

    if artery_range < total_slices * threshold and zf is not None and zb is not None:
        artery_bridged, _ = interpolate_circle_bridge(artery_mask, zf, zb)
        return extract_branches(artery_mask, artery_bridged, top_n=2)
    else:
        print("[WARN] Artery: gradient range exceeded - return zero array")
        return np.zeros_like(artery_mask)


def process_vein_branches(
    vein_mask: np.ndarray,
    z_start: int,
    z_end: int,
    percentile: int = VesselParams.VEIN_PERCENTILE,
    threshold: float = VesselParams.VEIN_THRESHOLD,
) -> np.ndarray:
    """
    정맥 분기 처리

    Args:
        vein_mask: 정맥 바이너리 마스크
        z_start: 분석 시작 Z (신장 범위)
        z_end: 분석 끝 Z (신장 범위)
        percentile: 그래디언트 임계값 백분위수
        threshold: 전체 슬라이스 대비 허용 비율

    Returns:
        신정맥(Renal_v) 분기 마스크
    """
    if not vein_mask.any():
        return np.zeros_like(vein_mask)

    total_slices = vein_mask.shape[0]
    zf, zb = detect_gradient_range(vein_mask, z_start, z_end, percentile=percentile)

    vein_range = zb - zf + 1 if (zf is not None and zb is not None) else total_slices
    print(f"[INFO] Vein  : index=[{zf}-{zb}], gradient range={vein_range}/{total_slices} ({vein_range/total_slices*100:.1f}%)")

    if vein_range < total_slices * threshold and zf is not None and zb is not None:
        vein_bridged = interpolate_ellipse_bridge(vein_mask, zf, zb)
        return extract_branches(vein_mask, vein_bridged, top_n=2)
    else:
        print("[WARN] Vein: gradient range exceeded - return zero array")
        return np.zeros_like(vein_mask)


def process_vessel_branches(
    label_array: np.ndarray,
    z_start: int,
    z_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    동맥/정맥 분기를 한 번에 처리

    Args:
        label_array: 세그멘테이션 라벨 배열
        z_start: 분석 시작 Z (신장 범위)
        z_end: 분석 끝 Z (신장 범위)

    Returns:
        (신동맥 분기 마스크, 신정맥 분기 마스크)
    """
    artery_mask = (label_array == Label.ARTERY).astype(np.uint8)
    vein_mask = (label_array == Label.VEIN).astype(np.uint8)

    renal_a = np.zeros_like(artery_mask)
    renal_v = np.zeros_like(vein_mask)

    try:
        # 동맥 처리
        if artery_mask.any():
            renal_a = process_artery_branches(artery_mask, z_start, z_end)
        else:
            print("[WARN] Artery: label not found, skipping.")

        # 정맥 처리
        if vein_mask.any():
            renal_v = process_vein_branches(vein_mask, z_start, z_end)
        else:
            print("[WARN] Vein: label not found, skipping.")

        return renal_a, renal_v

    except Exception as e:
        print(f"[ERROR] Vessel processing failed: {e} - return zero array")
        return renal_a, renal_v
