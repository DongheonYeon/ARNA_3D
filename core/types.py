"""
핵심 데이터 타입 정의

파이프라인 전반에서 사용되는 데이터 클래스를 정의합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import trimesh


@dataclass
class VolumeData:
    """
    NIfTI 볼륨 데이터 컨테이너

    SimpleITK 이미지의 핵심 정보를 담습니다.
    """
    array: np.ndarray  # (Z, Y, X) 순서
    spacing: tuple[float, float, float]  # (X, Y, Z) 순서 (SimpleITK 기본)
    origin: tuple[float, float, float]
    direction: tuple[float, ...]  # 9개 요소 (3x3 행렬 flatten)

    @property
    def shape(self) -> tuple[int, int, int]:
        """볼륨 shape (Z, Y, X)"""
        return self.array.shape

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        """spacing을 (Z, Y, X) 순서로 반환"""
        return (self.spacing[2], self.spacing[1], self.spacing[0])

    def get_label_mask(self, label_value: int) -> np.ndarray:
        """특정 라벨의 바이너리 마스크 반환"""
        return (self.array == label_value).astype(np.uint8)

    def has_label(self, label_value: int) -> bool:
        """특정 라벨이 존재하는지 확인"""
        return np.any(self.array == label_value)

    def get_label_z_range(self, label_value: int) -> tuple[int, int] | None:
        """특정 라벨이 존재하는 Z 범위 반환 (시작, 끝)"""
        mask = self.array == label_value
        z_exists = mask.any(axis=(1, 2))
        if not z_exists.any():
            return None
        indices = np.where(z_exists)[0]
        return int(indices[0]), int(indices[-1])


@dataclass
class MeshCollection:
    """
    메시 컬렉션 컨테이너

    여러 해부학적 구조물의 메시를 관리합니다.
    trimesh.Scene의 래퍼로, 추가적인 편의 기능을 제공합니다.
    """
    _meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)

    def add(self, name: str, mesh: trimesh.Trimesh) -> None:
        """메시 추가"""
        mesh.metadata["name"] = name
        self._meshes[name] = mesh

    def get(self, name: str) -> trimesh.Trimesh | None:
        """이름으로 메시 조회"""
        return self._meshes.get(name)

    def remove(self, name: str) -> trimesh.Trimesh | None:
        """메시 제거 및 반환"""
        return self._meshes.pop(name, None)

    def get_by_prefix(self, prefix: str) -> list[tuple[str, trimesh.Trimesh]]:
        """접두사로 시작하는 모든 메시 반환 (예: 'Kidney' → 'Kidney-L', 'Kidney-R')"""
        return [
            (name, mesh)
            for name, mesh in self._meshes.items()
            if name == prefix or name.startswith(f"{prefix}-")
        ]

    def names(self) -> list[str]:
        """모든 메시 이름 반환"""
        return list(self._meshes.keys())

    def items(self):
        """딕셔너리 items() 호환"""
        return self._meshes.items()

    def __iter__(self):
        return iter(self._meshes.items())

    def __len__(self) -> int:
        return len(self._meshes)

    def __contains__(self, name: str) -> bool:
        return name in self._meshes

    def to_scene(self) -> trimesh.Scene:
        """trimesh.Scene으로 변환"""
        scene = trimesh.Scene()
        for name, mesh in self._meshes.items():
            scene.add_geometry(mesh, node_name=name)
        return scene

    @classmethod
    def from_scene(cls, scene: trimesh.Scene) -> "MeshCollection":
        """trimesh.Scene에서 생성"""
        collection = cls()
        for name, mesh in scene.geometry.items():
            # metadata에서 이름 가져오기 (없으면 키 이름 사용)
            mesh_name = mesh.metadata.get("name", name)
            collection.add(mesh_name, mesh)
        return collection


@dataclass
class ProcessingContext:
    """
    파이프라인 처리 컨텍스트

    파이프라인 실행 중 필요한 모든 상태를 담습니다.
    """
    # 입력 데이터
    input_path: Path
    volume: VolumeData | None = None

    # 처리 결과
    meshes: MeshCollection = field(default_factory=MeshCollection)

    # 메타데이터
    case_id: str | None = None
    phase: str | None = None

    # 중간 결과 저장 (디버깅용)
    debug_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_path = Path(self.input_path)
