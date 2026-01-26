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
    
    case_num = "case_S007"
    input_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\mask\segment__combined.nii.gz"
    output_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\3d\obj_A.glb"
    result = core_smooth(input_path, output_path)