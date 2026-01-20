import os, sys, json, re, time
import SimpleITK as sitk
import trimesh
import pyvista as pv
from pathlib import Path
from threeDRecon import combineGLB2, processNii, processMesh

def parse_info(case_path):
    # id extraction
    case_pattern = r'case_([a-f0-9\-]+)'
    case_match = re.search(case_pattern, case_path)
    case_id = case_match.group(1) if case_match else None
    # phase extraction 
    filename = os.path.basename(case_path)
    phase_pattern = r'segment_([A-Z])'
    phase_match = re.search(phase_pattern, filename)
    case_phase =  phase_match.group(1) if phase_match else None
    return case_id, case_phase

# 디버그 저장 헬퍼
def save_debug(scene: trimesh.Scene, save_dir: str, phase: str, tag: str, debug: bool = False):
    """
    scene을 obj_{phase}_{tag}.glb로 저장 (debug=True일 때만)
    """
    if not debug:
        return None
    if scene is None:
        print(f"[WARN] Debug save skipped: scene is None ({tag})")
        return None
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"obj_{phase}_{tag}.glb")
    try:
        scene.export(out_path)
        print(f"[DEBUG] Saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARN] Debug save failed ({tag}): {e}")
        return None

def main(case_path,  debug=True):
    start_time = time.time()
    _, phase = parse_info(case_path)
    base_path = Path(case_path).parent.parent
    
    img = sitk.ReadImage(case_path)
    label_array = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    # spacing = img.GetSpacing()[::-1]  # (X, Y, Z) → (Z, Y, X)
    
    processed_img = processNii.preprocess(img, label_array)
    # processed_label_array = sitk.GetArrayFromImage(processed_img)
    # processed_spacing = processed_img.GetSpacing()[::-1]
    temp_path = Path(case_path).parent / f"temp.nii.gz"
    print(f"[INFO] Temporary NIfTI file created: {temp_path}")
    sitk.WriteImage(processed_img, temp_path)
    
    # ===== 라벨 1,2 → 2 =====
    arr = sitk.GetArrayFromImage(processed_img)
    arr[(arr == 1) | (arr == 2)] = 2
    kidney_img = sitk.GetImageFromArray(arr)
    kidney_img.CopyInformation(processed_img)
    temp_kidney_path = Path(case_path).parent / f"temp_kidney.nii.gz"
    sitk.WriteImage(kidney_img, temp_kidney_path)
    
    construct_glb = combineGLB2.combine_glb(temp_path, temp_kidney_path)
    
    save_dir = os.path.join(base_path, '3d')
    os.makedirs(save_dir, exist_ok=True)
    save_debug(construct_glb, save_dir, phase, "before_step1", debug)

    # ===== 생성된 임시 파일 삭제 =====
    try:
        if temp_path.exists():
            temp_path.unlink()
        if temp_kidney_path.exists():
            temp_kidney_path.unlink()
        print("[INFO] Temporary files removed.")
    except Exception as e:
        print(f"[WARN] Temp file cleanup failed: {e}")


    # 1st smoothing
    print("[INFO] Step1")
    new_scene = trimesh.Scene()
    with open(os.path.join("threeDRecon", "config", "parts_config1.json"), "r") as f:
        parts_config1 = json.load(f)
    for cfg in parts_config1:
        processMesh.mesh_smoothing(
            scene=construct_glb,
            new_scene=new_scene,
            part_name=cfg["name"],
            smoothing_func=cfg.get("smoothing_func"),
            smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
            dilation_func=cfg.get("dilation_func"),
            dilation_kwargs=cfg.get("dilation_kwargs", {}),
            simplification_func=cfg.get("simplification_func"),
            simplification_kwargs=cfg.get("simplification_kwargs", {})
        )
    save_debug(new_scene, save_dir, phase, "after_step1", debug)
    
    # poisson reconstruction
    poisson_recon = processMesh.process_poisson(new_scene)
    
    # 2nd smoothing
    print("[INFO] Step2")
    final_scene = trimesh.Scene()
    with open(os.path.join("threeDRecon", "config", "parts_config2.json"), "r") as f:
        parts_config2 = json.load(f)
    for cfg in parts_config2:
        processMesh.mesh_smoothing(
            scene=poisson_recon,
            new_scene=final_scene,
            part_name=cfg["name"],
            smoothing_func=cfg.get("smoothing_func"),
            smoothing_kwargs=cfg.get("smoothing_kwargs", {}),
            dilation_func=cfg.get("dilation_func"),
            dilation_kwargs=cfg.get("dilation_kwargs", {}),
            simplification_func=cfg.get("simplification_func"),
            simplification_kwargs=cfg.get("simplification_kwargs", {})
        )

    # save_dir = os.path.join(base_path, '3d')
    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'obj_{phase}.glb')
    final_scene.export(save_path)
    end_time = time.time()
    print(f"Process Done.\nExecution Time: {end_time - start_time:.2f} seconds")
    return save_path

if __name__ == "__main__":
    '''
    입력은 mask 경로로 받습니다.
    input = "path/case_0000/mask/segment_A.nii.gz"
    
    출력은 결과가 저장된 경로를 반환합니다.
    output = "path/case_0000/3d/obj_A.nii.gz"
    '''
    case_path = r".\data\case_N006\mask\segment_A.nii.gz"
    result = main(case_path)