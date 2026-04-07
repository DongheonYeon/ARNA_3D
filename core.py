from .pipeline import run_pipeline


def core_smooth(
    input_path: str,
    output_path: str,
    enable_vessel_branch_split: bool = True,
) -> str | None:
    return run_pipeline(
        input_path=input_path,
        output_path=output_path,
        debug=False,
        enable_vessel_branch_split=enable_vessel_branch_split,
    )
