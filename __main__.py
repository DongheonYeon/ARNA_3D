import argparse
from pathlib import Path
from pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="ARNA-3D: NIfTI 세그멘테이션을 3D GLB 모델로 변환",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="입력 NIfTI 파일 경로 (예: data/case_0001/mask/segment_A.nii.gz)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="출력 디렉토리 (기본: 입력 파일 상위의 3d/ 폴더)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="디버그 모드 (중간 결과 저장)",
    )

    args = parser.parse_args()

    result_path = run_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        debug=args.debug,
    )

    print(f"\n결과 파일: {result_path}")


if __name__ == "__main__":
    main()
