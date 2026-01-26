from .pipeline import run_pipeline


def core_smooth(input_path: str, output_path: str) -> str:
    return run_pipeline(
        input_path=input_path,
        output_path=output_path,
        debug=False,
    )