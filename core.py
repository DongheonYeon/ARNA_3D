from .pipeline import run_pipeline


def core_smooth(input_path: str, output_path: str) -> str:
    return run_pipeline(
        input_path=input_path,
        output_path=output_path,
        debug=False,
    )

if __name__ == "__main__":
    """
    실행 시 상위 디렉토리에서:
    python -m ARNA-3D.core
    """
    
    case_num = "test_006"
    input_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\mask\segment_A.nii.gz"
    output_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\3d\obj_A.glb"
    result = core_smooth(input_path, output_path)