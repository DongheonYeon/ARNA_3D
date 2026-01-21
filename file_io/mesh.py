"""
메시 파일 입출력 모듈

trimesh를 사용하여 GLB/OBJ 파일을 읽고 씁니다.
"""

from pathlib import Path
import trimesh

from config.logger import logger
from core.types import MeshCollection
from core.exceptions import MeshExtractionError


def load_mesh(file_path: Path | str) -> trimesh.Trimesh:
    """
    메시 파일 로드

    Args:
        file_path: 메시 파일 경로 (.glb, .obj 등)

    Returns:
        trimesh.Trimesh 객체

    Raises:
        MeshExtractionError: 로드 실패 시
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise MeshExtractionError(f"File not found: {file_path}")

    try:
        return trimesh.load(str(file_path))
    except Exception as e:
        raise MeshExtractionError(f"Fail to load mesh: {file_path} - {e}") from e


def save_mesh(mesh: trimesh.Trimesh, file_path: Path | str) -> Path:
    """
    단일 메시를 파일로 저장

    Args:
        mesh: 저장할 메시
        file_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(file_path))
    return file_path


def save_scene(scene: trimesh.Scene, file_path: Path | str) -> Path:
    """
    Scene을 파일로 저장

    Args:
        scene: 저장할 Scene
        file_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    scene.export(str(file_path))
    return file_path


def save_collection(collection: MeshCollection, file_path: Path | str) -> Path:
    """
    MeshCollection을 파일로 저장

    Args:
        collection: 저장할 MeshCollection
        file_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    scene = collection.to_scene()
    return save_scene(scene, file_path)


def save_debug_scene(
    scene: trimesh.Scene | None,
    save_dir: Path | str,
    phase: str,
    tag: str,
    debug: bool = False,
) -> Path | None:
    """
    디버그용 Scene 저장

    Args:
        scene: 저장할 Scene (None이면 무시)
        save_dir: 저장 디렉토리
        phase: phase 식별자 (예: 'A')
        tag: 태그 (예: 'after_step1')
        debug: False이면 저장하지 않음

    Returns:
        저장된 파일 경로 또는 None
    """
    if not debug:
        return None

    if scene is None:
        logger.warning(f"Skipping debug save: scene is None ({tag})")
        return None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"obj_{phase}_{tag}.glb"

    try:
        scene.export(str(out_path))
        logger.debug(f"Saved: {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"Failed to save debug file ({tag}): {e}")
        return None
