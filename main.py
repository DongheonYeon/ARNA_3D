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

if __name__ == "__main__":
    
    case_num = "case_S001"
    input_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\mask\segment__combined.nii.gz"
    output_path = fr"C:\Users\USER\Documents\Projects\ARNA-3D\data\{case_num}\3d\obj_A.glb"
    result = core_smooth(input_path, output_path)
