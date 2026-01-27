"""
메시 추출 모듈

NIfTI 볼륨에서 Marching Cubes로 메시를 추출합니다.
"""

from pathlib import Path
import vtk
from vtk.util.numpy_support import ( # type: ignore
    vtk_to_numpy,
    numpy_to_vtk,
    get_vtk_array_type,
)
import trimesh
import numpy as np

from ...config.constants import Label
from ...config.logger import logger
from ...domain.types import MeshCollection, VolumeData
from .splitting import split_bilateral, filter_valid_tumors
from .transform import rotate_and_center_scene


def _read_nifti_vtk(file_path: Path | str) -> vtk.vtkNIFTIImageReader:
    """VTK로 NIfTI 파일 읽기"""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(file_path))
    reader.Update()
    return reader


def _volume_to_vtk_image(volume: VolumeData) -> vtk.vtkImageData:
    """VolumeData를 VTK ImageData로 변환"""
    data = np.ascontiguousarray(volume.array)
    z_size, y_size, x_size = data.shape

    vtk_data = numpy_to_vtk(
        data.ravel(order="C"),
        deep=True,
        array_type=get_vtk_array_type(data.dtype),
    )

    image = vtk.vtkImageData()
    image.SetDimensions(x_size, y_size, z_size)
    image.SetSpacing(*volume.spacing)
    image.SetOrigin(*volume.origin)
    image.GetPointData().SetScalars(vtk_data)

    if hasattr(image, "SetDirectionMatrix") and len(volume.direction) == 9:
        direction = vtk.vtkMatrix3x3()
        for r in range(3):
            for c in range(3):
                direction.SetElement(r, c, volume.direction[r * 3 + c])
        image.SetDirectionMatrix(direction)

    return image


def _vtk_polydata_to_trimesh(polydata: vtk.vtkPolyData) -> trimesh.Trimesh | None:
    """VTK PolyData를 trimesh로 변환"""
    if not polydata or not polydata.GetPoints():
        return None

    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    polys = polydata.GetPolys()

    if not polys:
        return None

    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    faces = []

    while polys.GetNextCell(id_list):
        ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        if len(ids) == 3:
            faces.append(ids)
        elif len(ids) > 3:
            # 다각형 -> 삼각 팬으로 분할
            for i in range(1, len(ids) - 1):
                faces.append([ids[0], ids[i], ids[i + 1]])

    faces = np.array(faces)
    if len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=pts, faces=faces, process=False)


def _extract_single_label(
    source: vtk.vtkAlgorithm | vtk.vtkImageData,
    label_value: int,
) -> trimesh.Trimesh | None:
    """단일 라벨의 메시 추출"""
    extractor = vtk.vtkDiscreteMarchingCubes()
    if isinstance(source, vtk.vtkAlgorithm):
        extractor.SetInputConnection(source.GetOutputPort())
    else:
        extractor.SetInputData(source)
    extractor.SetValue(0, label_value)
    extractor.Update()

    polydata = extractor.GetOutput()
    if not polydata or polydata.GetNumberOfPolys() == 0:
        return None

    return _vtk_polydata_to_trimesh(polydata)


def extract_meshes_from_volume(
    nifti_data: VolumeData,
    kidney_nifti_data: VolumeData,
) -> MeshCollection:
    """
    NIfTI 볼륨에서 모든 라벨의 메시 추출

    Args:
        nifti_data: 메인 NIfTI 데이터
        kidney_nifti_data: 신장 전처리된 NIfTI 데이터

    Returns:
        MeshCollection 객체
    """
    source = _volume_to_vtk_image(nifti_data)
    source_kidney = _volume_to_vtk_image(kidney_nifti_data)
    collection = MeshCollection()

    for label in Label:
        label_name = label.name.capitalize()
        label_value = int(label)
        # 메시 추출
        base_mesh = _extract_single_label(source, label_value)

        if base_mesh is None or base_mesh.faces.size == 0:
            logger.warning(f"Skip {label_name}: label not found.")
            continue

        # 신장: L/R 분할
        if label_name == "Kidney" and source_kidney:
            kidney_mesh = _extract_single_label(source_kidney, label_value)
            if kidney_mesh:
                _add_bilateral_meshes(collection, "Kidney", kidney_mesh)
            continue

        # Fat: L/R 분할
        if label_name == "Fat":
            _add_bilateral_meshes(collection, "Fat", base_mesh)
            continue

        # Tumor: 유효성 검증 후 번호 부여
        if label_name == "Tumor":
            _add_tumor_meshes(collection, base_mesh)
            continue

        # 나머지: 그대로 추가
        collection.add(label_name, base_mesh)

    # 회전 및 중심 이동 적용
    scene = collection.to_scene()
    rotated_scene = rotate_and_center_scene(scene)

    return MeshCollection.from_scene(rotated_scene)


def _add_bilateral_meshes(
    collection: MeshCollection,
    base_name: str,
    mesh: trimesh.Trimesh,
) -> None:
    """좌/우 분할 메시 추가"""
    parts = split_bilateral(mesh)
    if not parts:
        return

    sides = ["L", "R"]
    for part, side in zip(parts, sides):
        part_name = f"{base_name}-{side}"
        collection.add(part_name, part)


def _add_tumor_meshes(
    collection: MeshCollection,
    mesh: trimesh.Trimesh,
) -> None:
    """Tumor 메시 추가 (유효성 검증 후 번호 부여)"""
    parts = mesh.split(only_watertight=False)
    if not parts:
        return

    valid_parts = filter_valid_tumors(parts)
    if not valid_parts:
        return

    # 크기순 정렬
    valid_parts = sorted(valid_parts, key=lambda m: len(m.faces), reverse=True)

    # 단일 tumor면 번호 없이, 여러 개면 번호 부여
    if len(valid_parts) == 1:
        collection.add("Tumor", valid_parts[0])
    else:
        for i, part in enumerate(valid_parts, start=1):
            part_name = f"Tumor-{i}"
            collection.add(part_name, part)
