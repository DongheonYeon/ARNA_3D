import importlib
import sys
import types
from pathlib import Path

package_name = "arna3d_pkg"
package_path = Path(__file__).resolve().parent

if package_name not in sys.modules:
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(package_path)]
    pkg.__file__ = str(package_path / "__init__.py")
    sys.modules[package_name] = pkg

run_pipeline = importlib.import_module(f"{package_name}.pipeline").run_pipeline


def core_smooth(input_path: str, output_path: str) -> str | None:
    return run_pipeline(
        input_path=input_path,
        output_path=output_path,
        debug=True,
    )
    
def process_all():
    data_dir = Path(__file__).resolve().parent / "data"
    case_dirs = sorted(data_dir.glob("case_*"))

    print(f"총 {len(case_dirs)}개의 케이스 발견")

    for i, case_dir in enumerate(case_dirs, 1):
        case_num = case_dir.name
        input_path = case_dir / "mask" / "segment__combined.nii.gz"
        output_path = case_dir / "3d" / "obj_A.glb"

        if not input_path.exists():
            print(f"[{i}/{len(case_dirs)}] {case_num}: 입력 파일 없음, 건너뜀")
            continue

        print(f"[{i}/{len(case_dirs)}] {case_num} 처리 중...")
        result = core_smooth(str(input_path), str(output_path))

        if result:
            print(f"[{i}/{len(case_dirs)}] {case_num} 완료: {result}")
        else:
            print(f"[{i}/{len(case_dirs)}] {case_num} 실패")

if __name__ == "__main__":

    # process_all()
    
    '''
    S022: 3d702647-5d73-4c93-85c4-566e5c587276 (큰 Tumor 2개, 리메싱 시간 측정용)
    S028: 048561bc-ae15-4458-9fc3-bf32c1a5f18d (안보이는 Tumor 하나 (신장외부), Artery 이상함)
    S018: 5df8e095-18e2-45b1-840e-5cde4979daac ()
    
    S006: 4a69f761-e9c5-46ab-8ff0-13bf2f418027 (너무 많은 Tumor)
    S002: aa507f5a-4b19-40d2-b706-1d424338ee77 (Vein 끊김)
    S001: d4178823-f496-4c53-9d3e-0aa403560adc
    '''
    case_num = "3d702647-5d73-4c93-85c4-566e5c587276"
    input_path = fr"C:\Users\USER\Documents\Projects\ARNA_3D\data\case_{case_num}\mask\segment__combined.nii.gz"
    output_path = fr"C:\Users\USER\Documents\Projects\ARNA_3D\data\case_{case_num}\3d\obj_A.glb"
    result = core_smooth(input_path, output_path)