import trimesh
import numpy as np
import pyvista as pv
import open3d as o3d
import manifold3d
import subprocess
import tempfile
import os


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


# Blender 실행 경로 (필요시 수정)
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

# Blender에서 실행할 Python 스크립트 템플릿
BLENDER_SCRIPT_TEMPLATE = '''
import bpy
import sys
import traceback

# 3D Print Toolbox 애드온 활성화 (Blender 5.0+에서 이름 변경됨)
PRINT3D_AVAILABLE = False
for mod_name in ["bl_ext.blender_org.print3d_toolbox", "object_print3d_utils"]:
    try:
        bpy.ops.preferences.addon_enable(module=mod_name)
        PRINT3D_AVAILABLE = True
        print(f"[Blender] 3D Print Toolbox enabled ({{mod_name}})")
        break
    except:
        continue

if not PRINT3D_AVAILABLE:
    print("[Blender] 3D Print Toolbox not available, using fallback")

def make_manifold(obj):
    """3D Print Toolbox의 Make Manifold로 watertight 메시 생성"""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    if PRINT3D_AVAILABLE:
        try:
            # 3D Print Toolbox의 Make Manifold (가장 효과적)
            bpy.ops.mesh.print3d_clean_non_manifold()
            print(f"[Blender] Make Manifold applied to {{obj.name}}")
        except Exception as e:
            print(f"[Blender] Make Manifold failed: {{e}}, using fallback")
            clean_mesh_fallback(obj)
    else:
        clean_mesh_fallback(obj)

    obj.select_set(False)

def clean_mesh_fallback(obj):
    """3D Print Toolbox 없을 때 기본 메시 정리"""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Fill holes
    try:
        bpy.ops.mesh.fill_holes(sides=0)
    except Exception as e:
        print(f"[Blender] fill_holes skipped: {{e}}")

    # 중복 버텍스 병합
    try:
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
    except Exception as e:
        print(f"[Blender] remove_doubles skipped: {{e}}")

    # Normals 통일
    try:
        bpy.ops.mesh.normals_make_consistent(inside=False)
    except Exception as e:
        print(f"[Blender] normals skipped: {{e}}")

    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

def boolean_difference(target_obj, cutter_obj):
    """Boolean difference 연산 수행"""
    bpy.context.view_layer.objects.active = target_obj
    target_obj.select_set(True)

    # Boolean modifier 추가
    bool_mod = target_obj.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = cutter_obj
    bool_mod.solver = 'EXACT'

    # Apply modifier
    try:
        bpy.ops.object.modifier_apply(modifier="Boolean")
        return True
    except Exception as e:
        print(f"[Blender] Boolean modifier apply failed: {{e}}")
        target_obj.modifiers.remove(bool_mod)
        return False
    finally:
        target_obj.select_set(False)

def decimate_mesh(obj, target_vertex_count):
    """Decimate modifier로 버텍스 수 축소"""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    current_verts = len(obj.data.vertices)
    if current_verts <= target_vertex_count:
        print(f"[Blender] {{obj.name}}: vertex count {{current_verts}} <= target {{target_vertex_count}}, skip decimation")
        obj.select_set(False)
        return

    # 목표 비율 계산
    ratio = target_vertex_count / current_verts

    # Decimate modifier 추가
    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate.decimate_type = 'COLLAPSE'
    decimate.ratio = ratio

    try:
        bpy.ops.object.modifier_apply(modifier="Decimate")
        new_verts = len(obj.data.vertices)
        print(f"[Blender] {{obj.name}}: decimated {{current_verts}} -> {{new_verts}} vertices")
    except Exception as e:
        print(f"[Blender] Decimate failed for {{obj.name}}: {{e}}")
        obj.modifiers.remove(decimate)

    obj.select_set(False)

try:
    # 메인 처리 시작
    print("[Blender] Starting mesh processing...")

    # 기존 오브젝트 삭제
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # GLB 파일 임포트
    input_path = r"{input_path}"
    print(f"[Blender] Importing: {{input_path}}")
    bpy.ops.import_scene.gltf(filepath=input_path)

    # 오브젝트들 분류
    kidney_objects = []
    tumor_objects = []
    other_objects = []

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            name = obj.name
            print(f"[Blender] Found mesh: {{name}}")
            if name.startswith("Kidney"):
                kidney_objects.append(obj)
            elif name.startswith("Tumor"):
                tumor_objects.append(obj)
            else:
                other_objects.append(obj)

    print(f"[Blender] Found {{len(kidney_objects)}} kidney, {{len(tumor_objects)}} tumor meshes")

    # Tumor가 없으면 그대로 저장
    if not tumor_objects:
        print("[Blender] No tumor found, exporting as-is")
        bpy.ops.export_scene.gltf(filepath=r"{output_path}", export_format='GLB')
        sys.exit(0)

    # Tumor들을 하나로 합치기
    if len(tumor_objects) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in tumor_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = tumor_objects[0]
        bpy.ops.object.join()
        combined_tumor = bpy.context.active_object
        combined_tumor.name = "Combined_Tumor"
    else:
        combined_tumor = tumor_objects[0]
        combined_tumor.name = "Combined_Tumor"

    # Tumor 메시를 watertight로 만들기
    print("[Blender] Processing tumor mesh...")
    make_manifold(combined_tumor)

    # 각 Kidney에 대해 Boolean 연산 수행
    for kidney_obj in kidney_objects:
        original_name = kidney_obj.name
        original_vertex_count = len(kidney_obj.data.vertices)
        print(f"[Blender] Processing {{original_name}} ({{original_vertex_count}} vertices)...")

        # Kidney 메시를 watertight로 만들기
        make_manifold(kidney_obj)

        # Boolean difference 수행
        success = boolean_difference(kidney_obj, combined_tumor)
        if success:
            print(f"[Blender] Boolean subtraction completed for {{original_name}}")
            # Boolean 후 버텍스 수 복원을 위해 Decimation 적용
            decimate_mesh(kidney_obj, original_vertex_count)
        else:
            print(f"[Blender] Boolean failed for {{original_name}}, using cleaned mesh")

        kidney_obj.name = original_name

    # Tumor 오브젝트 삭제 (Boolean에 사용된 후)
    bpy.data.objects.remove(combined_tumor, do_unlink=True)

    # 원래 Tumor들과 기타 메시들 다시 임포트
    print("[Blender] Re-importing tumor and other meshes...")
    bpy.ops.import_scene.gltf(filepath=input_path)

    # 새로 임포트된 오브젝트 중 Kidney는 제거 (이미 처리된 것 사용)
    for obj in list(bpy.context.selected_objects):
        if obj.type == 'MESH':
            # Kidney로 시작하면 제거 (처리된 버전 유지)
            if obj.name.startswith("Kidney"):
                bpy.data.objects.remove(obj, do_unlink=True)
            # Tumor나 기타 메시는 유지

    # 결과 내보내기
    output_path = r"{output_path}"
    print(f"[Blender] Exporting to: {{output_path}}")
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    print("[Blender] Done!")

except Exception as e:
    print(f"[Blender] ERROR: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''


def subtract_tumor_from_kidney_blender(
    scene: trimesh.Scene,
    blender_path: str = None
) -> trimesh.Scene:
    """
    Blender를 사용하여 Kidney 메시에서 Tumor를 Boolean Difference로 뺍니다.

    3D Print Toolbox 애드온의 Make Manifold 기능으로 메시를 정리한 후
    Boolean EXACT solver로 연산을 수행합니다.

    Args:
        scene: 입력 trimesh Scene
        blender_path: Blender 실행 파일 경로 (None이면 기본 경로 사용)

    Returns:
        Boolean 연산이 완료된 trimesh Scene
    """
    if blender_path is None:
        blender_path = BLENDER_PATH

    # Blender 존재 확인
    if not os.path.exists(blender_path):
        print(f"[WARN] Blender not found at {blender_path}, falling back to manifold3d")
        return subtract_tumor_from_kidney(scene)

    # Tumor 존재 확인
    tumor_exists = any(name.startswith("Tumor") for name in scene.geometry.keys())
    if not tumor_exists:
        print("[INFO] No tumor meshes found, skipping boolean subtraction")
        return scene

    # 임시 파일 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.glb")
        output_path = os.path.join(temp_dir, "output.glb")
        script_path = os.path.join(temp_dir, "blender_script.py")

        # 입력 Scene을 GLB로 저장
        scene.export(input_path)

        # Blender 스크립트 생성
        script_content = BLENDER_SCRIPT_TEMPLATE.format(
            input_path=input_path.replace("\\", "\\\\"),
            output_path=output_path.replace("\\", "\\\\")
        )

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Blender 실행
        print(f"[INFO] Running Blender for boolean subtraction...")
        try:
            result = subprocess.run(
                [blender_path, "--background", "--python", script_path],
                capture_output=True,
                text=True,
                timeout=120  # 2분 타임아웃
            )

            # Blender 출력 표시 (디버그용 전체 출력)
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.startswith("[Blender]") or "Error" in line or "error" in line:
                        print(f"       {line}")

            if result.stderr:
                # Python traceback이나 에러 메시지 출력
                stderr_lines = result.stderr.split('\n')
                error_lines = [l for l in stderr_lines if l.strip() and not l.startswith("Read")]
                if error_lines:
                    print(f"[WARN] Blender stderr:")
                    for line in error_lines[-10:]:  # 마지막 10줄만
                        print(f"       {line}")

            if result.returncode != 0:
                print(f"[WARN] Blender exited with code {result.returncode}")
                return scene

            # 결과 로드
            if os.path.exists(output_path):
                result_scene = trimesh.load(output_path)

                # trimesh.load가 Scene이 아닌 경우 처리
                if isinstance(result_scene, trimesh.Trimesh):
                    new_scene = trimesh.Scene()
                    new_scene.add_geometry(result_scene)
                    result_scene = new_scene

                print(f"[INFO] Boolean subtraction completed via Blender")
                return result_scene
            else:
                print(f"[WARN] Blender output file not found")
                return scene

        except subprocess.TimeoutExpired:
            print(f"[WARN] Blender timed out, using original scene")
            return scene
        except Exception as e:
            print(f"[WARN] Blender execution failed: {e}")
            return scene