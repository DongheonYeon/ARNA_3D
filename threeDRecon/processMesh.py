import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d
import manifold3d


def default_dilation(mesh: pv.PolyData, offset: float = 0.5) -> pv.PolyData:
    mesh.compute_normals(inplace=True)
    mesh.points += offset * mesh.point_normals
    return mesh

def mesh_smoothing(
    scene: trimesh.Scene,
    new_scene: trimesh.Scene,
    part_name: str,
    smoothing_func: str = None,
    smoothing_kwargs: dict = None,
    dilation_func: str = None,
    dilation_kwargs: dict = None,
    simplification_func: str = None, 
    simplification_kwargs: dict = None
    ):
    SMOOTHING_FUNC_MAP = {
        "laplacian": pv.PolyData.smooth,
        "taubin": pv.PolyData.smooth_taubin,
        None: lambda mesh, **kwargs: mesh  # None일 경우 아무 처리도 하지 않음
    }
    DILATION_FUNC_MAP = {
        "default": default_dilation,
        None: lambda mesh, **kwargs: mesh  # None일 경우 아무 처리도 하지 않음
    }
    SIMPLIFICATION_FUNC_MAP = {
        "decimate": pv.PolyData.decimate,      
        "decimate_pro": pv.PolyData.decimate_pro, 
        None: lambda mesh, **kwargs: mesh
    }

    # part_name과 정확히 일치하거나 "part_name-" 으로 시작하는 모든 geometry 찾기
    matching_names = [name for name in scene.geometry.keys()
                      if name == part_name or name.startswith(f"{part_name}-")]

    if not matching_names:
        print(f"[WARN] Skipping {part_name}: mesh not found in scene.")
        return

    # 매칭된 각 geometry에 대해 처리
    for matched_name in matching_names:
        mesh = scene.geometry.get(matched_name)
        if mesh is None:
            continue

        _process_single_mesh(
            mesh, matched_name, new_scene,
            smoothing_func, smoothing_kwargs,
            dilation_func, dilation_kwargs,
            simplification_func, simplification_kwargs,
            SMOOTHING_FUNC_MAP, DILATION_FUNC_MAP, SIMPLIFICATION_FUNC_MAP
        )

def _process_single_mesh(
    mesh, part_name, new_scene,
    smoothing_func, smoothing_kwargs,
    dilation_func, dilation_kwargs,
    simplification_func, simplification_kwargs,
    SMOOTHING_FUNC_MAP, DILATION_FUNC_MAP, SIMPLIFICATION_FUNC_MAP
    ):
    print(f"[INFO] Processing {part_name}")
    # Convert to PyVista PolyData: faces=[n, v0, v1, v2, ...]
    vertices = mesh.vertices
    faces = np.hstack([[3, *f] for f in mesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    # Dilation
    if dilation_func:
        dilation_fn = DILATION_FUNC_MAP.get(dilation_func)
        if dilation_fn is None:
            raise ValueError(f"[ERROR] Unknown dilation_func: {dilation_func}")
        print(f"{'':7}- Apply Dilation")
        pv_mesh = dilation_fn(pv_mesh, **(dilation_kwargs or {}))

    # Smoothing
    if smoothing_func:
        smoothing_fn = SMOOTHING_FUNC_MAP.get(smoothing_func)
        if smoothing_fn is None:
            raise ValueError(f"[ERROR] Unknown smoothing_func: {smoothing_func}")
        print(f"{'':7}- Apply Smoothing")
        pv_mesh = smoothing_fn(pv_mesh, **(smoothing_kwargs or {}))

    # Simplification (by Decimation)
    if simplification_func:
        simp_method = SIMPLIFICATION_FUNC_MAP.get(simplification_func)
        if simp_method is None:
             print(f"[WARN] Unknown simplification type: {simplification_func}")
        else:
            # target_reduction: 0.0 ~ 1.0 (예: 0.5면 면의 50%를 제거)
            print(f"{'':7}- Apply Simplification ({simplification_func})")
            pv_mesh = simp_method(pv_mesh, **(simplification_kwargs or {}))
            # pv_mesh.clean(tolerance=1.0, inplace=True)
            # pv_mesh = pv_mesh.triangulate()

    # Convert back to trimesh
    final_mesh = trimesh.Trimesh(
        vertices=pv_mesh.points,
        faces=pv_mesh.faces.reshape(-1, 4)[:, 1:]
    )
    final_mesh.metadata["name"] = part_name
    new_scene.add_geometry(final_mesh, node_name=part_name)

def poisson_reconstruction(mesh_list, depth=8):
    if not mesh_list:
        raise ValueError("[ERROR] Input mesh_list empty")
    
    print(f"{'':7}- Merged mesh: {len(mesh_list)}")

    # 병합
    merged = trimesh.util.concatenate(mesh_list)

    # Trimesh → Open3D 변환
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(merged.vertices),
        triangles=o3d.utility.Vector3iVector(merged.faces)
    )
    o3d_mesh.compute_vertex_normals()

    # 포인트 샘플링 (40000)
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=40000)
    pcd.estimate_normals()

    # Poisson 재구성
    mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh_out = mesh_out.crop(pcd.get_axis_aligned_bounding_box())

    # 다시 Trimesh로 변환
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_out.vertices),
        faces=np.asarray(mesh_out.triangles)
    )

    # Poisson 결과에 대해 LCC 적용
    components = tri_mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"{'':7}- Connected components: {len(components)} - using largest")
        tri_mesh = max(components, key=lambda c: c.area)

    return tri_mesh

def process_poisson(scene):
    # 이름 → mesh 매핑
    name_to_mesh = {mesh.metadata.get("name", k): mesh for k, mesh in scene.geometry.items()}

    # 병합 대상 그룹 정의
    artery_group = ["Artery", "Renal_a"]
    vein_group = ["Vein", "Renal_v"]

    # 결과 scene
    rec_scene = trimesh.Scene()

    # Artery (Aretry + Renal_a)
    artery_meshes = [name_to_mesh[name] for name in artery_group if name in name_to_mesh]
    if artery_meshes:
        print("[INFO] Processing Artery group")
        smoothed_artery = poisson_reconstruction(artery_meshes, depth=8)
        smoothed_artery.metadata["name"] = "Artery"
        rec_scene.add_geometry(smoothed_artery, node_name="Artery")
    
    # Artery (Artery + Renal_a) without Poisson
    # artery_meshes = [name_to_mesh[name] for name in artery_group if name in name_to_mesh]
    # if artery_meshes:
    #     print("[INFO] Merging Artery group (Artery + Renal_a)")
    #     if len(artery_meshes) > 1:
    #         merged_artery = trimesh.util.concatenate(artery_meshes)
    #     else:
    #         merged_artery = artery_meshes[0].copy()
    #     merged_artery.metadata["name"] = "Artery"
    #     rec_scene.add_geometry(merged_artery, node_name="Artery")

    # Vein System (Vein + Renal_v)
    vein_meshes = [name_to_mesh[name] for name in vein_group if name in name_to_mesh]
    if vein_meshes:
        print("[INFO] Processing Vein group")
        smoothed_vein = poisson_reconstruction(vein_meshes, depth=8)
        smoothed_vein.metadata["name"] = "Vein"
        rec_scene.add_geometry(smoothed_vein, node_name="Vein")

    # 나머지 구조물은 그대로 추가
    excluded = set(artery_group + vein_group)
    # excluded = set(vein_group)
    for name, mesh in name_to_mesh.items():
        if name not in excluded:
            mesh.name = name
            mesh.metadata["name"] = name
            rec_scene.add_geometry(mesh, node_name=name)

    return rec_scene


def trimesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """trimesh를 manifold3d 객체로 변환"""
    mesh_data = manifold3d.Mesh(
        vert_properties=np.array(mesh.vertices, dtype=np.float32),
        tri_verts=np.array(mesh.faces, dtype=np.uint32)
    )
    return manifold3d.Manifold(mesh_data)


def manifold_to_trimesh(manifold: manifold3d.Manifold) -> trimesh.Trimesh:
    """manifold3d 객체를 trimesh로 변환"""
    mesh_data = manifold.to_mesh()
    vertices = mesh_data.vert_properties[:, :3]  # xyz 좌표만 추출
    faces = mesh_data.tri_verts
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def subtract_tumor_from_kidney(scene: trimesh.Scene) -> trimesh.Scene:
    """
    Kidney 메시에서 Tumor 메시를 Boolean Difference로 빼서
    Z-fighting과 가림 현상을 방지합니다.

    스무딩이 완료된 후 호출해야 합니다.
    """
    # scene에서 geometry 추출
    geometry_dict = dict(scene.geometry)

    # Tumor 메시들 수집 (Tumor-1, Tumor-2, ... 형태)
    tumor_meshes = []
    tumor_names = []
    for name, mesh in geometry_dict.items():
        if name.startswith("Tumor"):
            tumor_meshes.append(mesh)
            tumor_names.append(name)

    if not tumor_meshes:
        print("[INFO] No tumor meshes found, skipping boolean subtraction")
        return scene

    # Tumor 메시들을 하나로 병합
    if len(tumor_meshes) == 1:
        combined_tumor = tumor_meshes[0]
    else:
        combined_tumor = trimesh.util.concatenate(tumor_meshes)

    print(f"[INFO] Boolean subtraction: {len(tumor_meshes)} tumor(s) from kidney")

    # Tumor를 manifold로 변환
    try:
        tumor_manifold = trimesh_to_manifold(combined_tumor)
    except Exception as e:
        print(f"[WARN] Failed to convert tumor to manifold: {e}")
        return scene

    # Kidney 메시들 처리 (Kidney-L, Kidney-R)
    new_scene = trimesh.Scene()

    for name, mesh in geometry_dict.items():
        if name.startswith("Kidney"):
            try:
                # Boolean difference: Kidney - Tumor
                kidney_manifold = trimesh_to_manifold(mesh)
                result_manifold = kidney_manifold - tumor_manifold

                if result_manifold.is_empty():
                    print(f"[WARN] Boolean difference result empty for {name}, using original mesh")
                    result = mesh
                else:
                    result = manifold_to_trimesh(result_manifold)
                    print(f"[INFO] Boolean subtraction completed for {name}")

                result.metadata["name"] = name
                new_scene.add_geometry(result, node_name=name)
            except Exception as e:
                print(f"[WARN] Boolean operation failed for {name}: {e}, using original mesh")
                mesh.metadata["name"] = name
                new_scene.add_geometry(mesh, node_name=name)
        else:
            # Kidney가 아닌 메시는 그대로 추가
            mesh.metadata["name"] = name
            new_scene.add_geometry(mesh, node_name=name)

    return new_scene