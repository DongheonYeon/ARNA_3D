from pipeline import run_pipeline


def core_smooth(input_path: str, output_path: str) -> str:
    return run_pipeline(
        input_path=input_path,
        output_path=output_path,
        debug=False,
    )

if __name__ == "__main__":
    result = core_smooth(r"./data/test_006/mask/segment_A.nii.gz", r"./data/test_006/3d/obj_A.glb")