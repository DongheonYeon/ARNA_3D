import os
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import trimesh

# ----------------------------------------------------------------------------
# 라벨 이름 매핑
# ----------------------------------------------------------------------------
LABELS = {
    "Tumor": 1,
    "Kidney": 2,
    "Artery": 3,
    "Vein": 4,
    "Ureter": 5,
    "Fat": 6,
    "Renal_a": 7,
    "Renal_v": 8
}

def read_volume(file_name):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader

def create_mask_extractor(nifti_reader):
    extractor = vtk.vtkDiscreteMarchingCubes()
    extractor.SetInputConnection(nifti_reader.GetOutputPort())
    return extractor

def create_polygon_reducer(input_connection, reduction=0.0):
    reducer = vtk.vtkDecimatePro()
    reducer.SetInputConnection(input_connection)
    reducer.SetTargetReduction(reduction)
    reducer.PreserveTopologyOn()
    return reducer

def vtk_polydata_to_trimesh(polydata):
    if not polydata or not polydata.GetPoints():
        return None

    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    polys = polydata.GetPolys()
    if not polys:
        return None
    
    # if spacing:
    #     pts *= np.array(spacing, dtype=np.float32)  # 물리 좌표 반영

    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    faces = []
    while polys.GetNextCell(id_list):
        ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        if len(ids) == 3:
            faces.append(ids)
        elif len(ids) > 3:
            # 다각형 -> 삼각 팬으로 분할
            for i in range(1, len(ids)-1):
                faces.append([ids[0], ids[i], ids[i+1]])
    faces = np.array(faces)
    if len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=pts, faces=faces, process=False)

def rotate_and_center(scene):
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    
    # 중심 이동 행렬
    T = trimesh.transformations.translation_matrix(-center)
    
    # 회전 행렬
    angle_rad = np.pi / 2
    R1 = trimesh.transformations.rotation_matrix(angle_rad, direction=[-1, 0, 0], point=center)

    # 변환
    combined_transform = trimesh.transformations.concatenate_matrices(T, R1)
    scene.apply_transform(combined_transform)
    
    # 변환을 메시 데이터에 직접 적용
    transformed_geometries = scene.dump()
    new_scene = trimesh.Scene(transformed_geometries)
    return new_scene

def make_glb(temp_path, temp_kidney_path):
    reader = read_volume(temp_path)
    reader_kidney = read_volume(temp_kidney_path)
    scene = trimesh.Scene()

    for label_name, label_val in LABELS.items():        
        # 1) Marching Cubes
        extractor = create_mask_extractor(reader)
        extractor.SetValue(0, label_val)
        extractor.Update()
        polydata = extractor.GetOutput()

        extractor_kidney = create_mask_extractor(reader_kidney)
        extractor_kidney.SetValue(0, label_val)
        extractor_kidney.Update()
        polydata_kidney = extractor_kidney.GetOutput()

        if not polydata or polydata.GetNumberOfPolys() == 0:
            print(f"[WARN] '{label_name}' 빈 메쉬, skip")
            continue

        # # 2) Decimate
        # reducer = create_polygon_reducer(extractor.GetOutputPort(), reduction=0.0)
        # reducer.Update()
        # reduced_poly = reducer.GetOutput()
        # if not reduced_poly or reduced_poly.GetNumberOfPolys() == 0:
        #     continue

        base_mesh = vtk_polydata_to_trimesh(polydata)
        kidney_mesh = vtk_polydata_to_trimesh(polydata_kidney)
        if base_mesh is None or base_mesh.faces.size == 0:
            print(f"[WARN] '{label_name}' 변환 실패, skip")
            continue
        
        if label_name == "Kidney":
            parts = kidney_mesh.split(only_watertight=False)
            if not parts:
                continue
            parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)[:2]
            parts = sorted(parts, key=lambda m: m.centroid[2])
            sides = ["L", "R"]
            for part, side in zip(parts, sides):
                part_name = f"{label_name}-{side}"
                part.metadata["name"] = part_name
                scene.add_geometry(part, node_name=part_name)
        elif label_name == "Fat":
            parts = base_mesh.split(only_watertight=False)
            if not parts:
                continue
            parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)[:2]
            parts = sorted(parts, key=lambda m: m.centroid[2])
            sides = ["L", "R"]
            for part, side in zip(parts, sides):
                part_name = f"{label_name}-{side}"
                part.metadata["name"] = part_name
                scene.add_geometry(part, node_name=part_name)
        elif label_name == "Tumor":
            parts = base_mesh.split(only_watertight=False)
            if not parts:
                continue
            # 내부 빈 공간 메시 제거 (음수 볼륨 또는 다른 메시에 포함된 경우)
            valid_parts = []
            for part in parts:
                # 볼륨이 음수면 내부 공간 (노멀이 안쪽을 향함)
                if part.is_watertight and part.volume < 0:
                    print(f"[INFO] Tumor 내부 공간 제거 (음수 볼륨: {part.volume:.2f})")
                    continue
                valid_parts.append(part)

            if not valid_parts:
                continue
            valid_parts = sorted(valid_parts, key=lambda m: len(m.faces), reverse=True)
            for i, part in enumerate(valid_parts, start=1):
                part_name = f"{label_name}-{i}"
                part.metadata["name"] = part_name
                scene.add_geometry(part, node_name=part_name)
        else:
            base_mesh.metadata["name"] = label_name
            scene.add_geometry(base_mesh, node_name=label_name)
    
    scene = rotate_and_center(scene)
    return scene

def combine_glb(temp_path, temp_kidney_path):
    print(f"[INFO] 변환 중: {temp_path}")
    result = make_glb(temp_path, temp_kidney_path)
    # result.export(r"C:\Users\USER\Documents\vscode_projects\ARNA-3D\data\case_0001\mask\temp.obj")
    print(f"[INFO] 결과: {result}")
    return result

if __name__ == "__main__":
    temp_path = r"C:\Users\USER\Documents\vscode_projects\ARNA-3D\data\case_0001\mask\temp.nii.gz"
    combine_glb(temp_path)
