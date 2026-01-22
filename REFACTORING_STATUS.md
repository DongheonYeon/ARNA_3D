# ARNA-3D 리팩토링 진행상황

> 작성일: 2026-01-21
> 목적: 코드베이스 완전 리팩토링 - 로직 차용, 구조 개선

---

## 1. 프로젝트 배경

### 원본 코드의 문제점

**단일 책임 원칙(SRP) 위반:**

- `processNii.py`의 `process_vessels()`: 동맥 처리 + 정맥 처리 + 예외 처리를 한 함수에서 담당
- `combineGLB2.py`의 `make_glb()`: 메시 추출 + 분할(L/R) + 검증(Tumor) + 회전 변환 혼합
- `processMesh.py`의 `_process_single_mesh()`: 11개 인자, Dilation + Smoothing + Simplification 혼합
- `main.py`의 `main()`: 전처리 + 임시파일 관리 + GLB 생성 + 스무딩(2단계) + 저장

**기타 문제:**

- 매직 넘버 남발 (percentile=95, depth=8, 40000 등 하드코딩)
- 중복 코드 (Kidney/Fat L/R 분할 로직, combineGLB.py vs combineGLB2.py)
- 임시 파일 관리 취약 (부모 디렉토리에 temp.nii.gz 생성)
- 설정 경로 하드코딩 (`os.path.join("threeDRecon", "config", ...)`)
- 일관되지 않은 에러 처리

### 리팩토링 목표

1. 단일 책임 원칙 준수 (함수당 하나의 역할)
2. 모듈 경계 재설정 (기능 단위 분리)
3. 상수/설정 중앙 관리
4. 타입 힌팅 전체 적용
5. 임시 파일 안전 관리 (context manager)
6. 향후 멀티프로세싱 도입 가능한 구조 (병렬화는 이번 작업에서 제외)

---

## 2. 완료된 작업

### 2.1 새 패키지 구조 생성 (`arna_3d/`)

```
arna_3d/
├── __init__.py              # 패키지 초기화, __version__ 정의
├── __main__.py              # CLI 진입점 (argparse)
│
├── config/                  # [완료] 설정 및 상수
│   ├── __init__.py
│   ├── constants.py         # Label(IntEnum), VesselParams, PoissonParams, MorphologyParams
│   ├── settings.py          # PipelineSettings, SmoothingConfig, SmoothingPreset, load_smoothing_preset()
│   └── presets/
│       ├── stage1.json      # 1단계 스무딩 설정 (기존 parts_config1.json 복사)
│       └── stage2.json      # 2단계 스무딩 설정 (기존 parts_config2.json 복사)
│
├── domain/                  # [완료] 핵심 데이터 타입
│   ├── __init__.py
│   └── types.py             # VolumeData, MeshCollection, ProcessingContext
│
├── file_io/                 # [완료] 입출력
│   ├── __init__.py
│   ├── nifti.py             # load_nifti(), save_nifti(), volume_to_sitk(), copy_metadata()
│   ├── mesh.py              # load_mesh(), save_mesh(), save_scene(), save_debug_scene()
│   └── temp.py              # TempFileManager (context manager), temp_nifti_file()
│
├── threeDrecon/             # [완료] 처리 로직
│   ├── __init__.py
│   ├── segmentation/        # 세그멘테이션 전처리
│   │   ├── __init__.py
│   │   ├── preprocessing.py # preprocess_segmentation(), apply_fat_dilation(), preprocess_kidney_segmentation()
│   │   └── morphology.py    # get_largest_component()
│   │
│   ├── vessel/              # 혈관 분석 (기존 processNii.py에서 분리)
│   │   ├── __init__.py
│   │   ├── analysis.py      # compute_radii_array(), detect_gradient_range(), detect_zscore_range()
│   │   ├── interpolation.py # interpolate_circle_bridge(), interpolate_ellipse_bridge()
│   │   └── branch.py        # extract_branches(), process_artery_branches(), process_vein_branches()
│   │
│   └── mesh/                # 메시 처리 (기존 combineGLB2.py, processMesh.py에서 분리)
│       ├── __init__.py
│       ├── extraction.py    # extract_meshes_from_volume() - VTK Marching Cubes
│       ├── splitting.py     # split_bilateral(), filter_valid_tumors()
│       ├── smoothing.py     # apply_smoothing(), apply_dilation(), smooth_mesh_collection()
│       ├── reconstruction.py # poisson_reconstruct(), process_vessel_reconstruction()
│       ├── transform.py     # rotate_and_center_scene(), rotate_and_center_mesh()
│       └── conversion.py    # trimesh_to_pyvista(), pyvista_to_trimesh(), trimesh_to_open3d()
│
└── runner.py                # [완료] Pipeline 클래스, run_pipeline() 편의 함수
```

### 2.2 핵심 데이터 클래스

```python
# arna_3d/domain/types.py

@dataclass
class VolumeData:
    """NIfTI 볼륨 데이터 컨테이너"""
    array: np.ndarray          # (Z, Y, X) 순서
    spacing: tuple[float, ...]  # (X, Y, Z) 순서
    origin: tuple[float, ...]
    direction: tuple[float, ...]

@dataclass
class MeshCollection:
    """trimesh.Scene 래퍼"""
    _meshes: dict[str, trimesh.Trimesh]

    def add(name, mesh): ...
    def get(name): ...
    def get_by_prefix(prefix): ...  # 'Kidney' → ['Kidney-L', 'Kidney-R']
    def to_scene(): ...
    @classmethod
    def from_scene(scene): ...
```

### 2.3 상수 정의

```python
# arna_3d/config/constants.py

class Label(IntEnum):
    TUMOR = 1
    KIDNEY = 2
    ARTERY = 3
    VEIN = 4
    URETER = 5
    FAT = 6
    RENAL_A = 7
    RENAL_V = 8

@dataclass(frozen=True)
class VesselParams:
    ARTERY_PERCENTILE: int = 95
    ARTERY_THRESHOLD: float = 0.5
    VEIN_PERCENTILE: int = 90
    VEIN_THRESHOLD: float = 0.7
    DILATION_ITERATIONS: int = 3

@dataclass(frozen=True)
class PoissonParams:
    DEPTH: int = 8
    SAMPLE_POINTS: int = 40000
```

### 2.4 파이프라인 실행 흐름

```python
# arna_3d/runner.py

class Pipeline:
    def run(self) -> Path:
        # 1. NIfTI 로드
        volume = load_nifti(input_path)

        # 2. 세그멘테이션 전처리 (혈관 분기 분할, Fat dilation)
        processed = preprocess_segmentation(volume)

        # 3. 신장용 볼륨 생성 (Tumor → Kidney 병합)
        kidney_volume = preprocess_kidney_segmentation(processed)

        # 4. 임시 파일로 저장 후 메시 추출
        with TempFileManager() as temp_mgr:
            temp_nifti = temp_mgr.save_volume_temp(processed)
            temp_kidney = temp_mgr.save_volume_temp(kidney_volume)
            meshes = extract_meshes_from_volume(temp_nifti, temp_kidney)

        # 5. 1단계 스무딩
        meshes = smooth_mesh_collection(meshes, stage1_preset)

        # 6. Poisson 재구성 (Artery+Renal_a, Vein+Renal_v)
        meshes = process_vessel_reconstruction(meshes)

        # 7. 2단계 스무딩
        meshes = smooth_mesh_collection(meshes, stage2_preset)

        # 8. 결과 저장
        return save_scene(meshes.to_scene(), output_path)
```

---

## 3. 미완료 작업

### 3.1 테스트 실행 (우선순위: 높음)

새 패키지가 실제로 동작하는지 검증 필요:

```bash
python -m arna_3d ./data/case_S004/mask/segment_A.nii.gz --debug
```

**예상되는 문제:**

- import 경로 오류 가능성
- 함수 시그니처 불일치 가능성
- 기존 로직과의 미세한 차이

### 3.2 기존 코드 제거 (우선순위: 중간)

테스트 완료 후 제거할 파일:

- `main.py`
- `threeDRecon/processNii.py`
- `threeDRecon/combineGLB.py` (이미 미사용)
- `threeDRecon/combineGLB2.py`
- `threeDRecon/processMesh.py`
- `threeDRecon/config/` 폴더 (프리셋은 새 위치로 이동 완료)

### 3.3 추가 개선 사항 (우선순위: 낮음)

- 로깅 시스템 도입 (`print` → `logging` 모듈)
- 단위 테스트 작성
- GitHub Actions CI/CD 설정
- 성능 프로파일링 및 최적화

---

## 4. 기존 코드 ↔ 새 코드 매핑

| 기존 파일        | 기존 함수                     | 새 위치                                     | 새 함수                           |
| ---------------- | ----------------------------- | ------------------------------------------- | --------------------------------- |
| `processNii.py`  | `get_largest_component()`     | `threeDrecon/vessel/analysis.py`            | `get_largest_component()`         |
| `processNii.py`  | `get_radii_array()`           | `threeDrecon/vessel/analysis.py`            | `compute_radii_array()`           |
| `processNii.py`  | `get_gradient_range()`        | `threeDrecon/vessel/analysis.py`            | `detect_gradient_range()`         |
| `processNii.py`  | `get_zscore_range()`          | `threeDrecon/vessel/analysis.py`            | `detect_zscore_range()`           |
| `processNii.py`  | `interpolate_circle_bridge()` | `threeDrecon/vessel/interpolation.py`       | `interpolate_circle_bridge()`     |
| `processNii.py`  | `interpolate_vein()`          | `threeDrecon/vessel/interpolation.py`       | `interpolate_ellipse_bridge()`    |
| `processNii.py`  | `process_vessels()`           | `threeDrecon/vessel/branch.py`              | `process_vessel_branches()`       |
| `processNii.py`  | `preprocess()`                | `threeDrecon/segmentation/preprocessing.py` | `preprocess_segmentation()`       |
| `combineGLB2.py` | `make_glb()`                  | `threeDrecon/mesh/extraction.py`            | `extract_meshes_from_volume()`    |
| `combineGLB2.py` | `rotate_and_center()`         | `threeDrecon/mesh/transform.py`             | `rotate_and_center_scene()`       |
| `processMesh.py` | `mesh_smoothing()`            | `threeDrecon/mesh/smoothing.py`             | `smooth_mesh_collection()`        |
| `processMesh.py` | `_process_single_mesh()`      | `threeDrecon/mesh/smoothing.py`             | `process_single_mesh()`           |
| `processMesh.py` | `poisson_reconstruction()`    | `threeDrecon/mesh/reconstruction.py`        | `poisson_reconstruct()`           |
| `processMesh.py` | `process_poisson()`           | `threeDrecon/mesh/reconstruction.py`        | `process_vessel_reconstruction()` |
| `main.py`        | `main()`                      | `runner.py`                                 | `Pipeline.run()`                  |

---

## 5. 주의사항

### 5.1 로직 변경 없음 원칙

리팩토링 시 기존 로직을 그대로 유지했습니다. 다음 항목들은 의도적으로 변경하지 않음:

- 혈관 분석 알고리즘 (그래디언트 기반 범위 탐지)
- 스무딩 파라미터
- 좌표계 변환 (NIfTI → GLB)
- L/R 분할 기준 (Z축 centroid)

### 5.2 의존성

기존 `requirements.txt`와 동일한 패키지 사용:

- SimpleITK: NIfTI I/O
- VTK: Marching Cubes
- trimesh: 메시 조작
- pyvista: 스무딩
- open3d: Poisson 재구성
- numpy, scipy, scikit-image, opencv-python: 수치 연산

### 5.3 Python 버전

타입 힌팅에 Python 3.10+ 문법 사용:

```python
def func() -> tuple[int, int] | None:  # 3.10+
```

Python 3.9 이하에서는 `from __future__ import annotations` 추가 필요.

---

## 6. 다음 세션에서 할 일

1. **테스트 실행**

   ```bash
   cd c:\Users\USER\Documents\Projects\ARNA-3D
   python -m arna_3d ./data/test_006/mask/segment_A.nii.gz --debug
   ```

2. **오류 수정**: import 오류, 타입 오류 등 발생 시 수정

3. **결과 비교**: 기존 `main.py` 출력물과 새 패키지 출력물 비교

4. **기존 코드 제거**: 테스트 통과 후 레거시 파일 삭제

5. **커밋**: 리팩토링 완료 커밋

---

## 7. 파일 목록 (새로 생성됨)

```
arna_3d/__init__.py
arna_3d/__main__.py
arna_3d/config/__init__.py
arna_3d/config/constants.py
arna_3d/config/settings.py
arna_3d/config/presets/stage1.json
arna_3d/config/presets/stage2.json
arna_3d/domain/__init__.py
arna_3d/domain/types.py
arna_3d/file_io/__init__.py
arna_3d/file_io/mesh.py
arna_3d/file_io/nifti.py
arna_3d/file_io/temp.py
arna_3d/runner.py
arna_3d/threeDrecon/__init__.py
arna_3d/threeDrecon/mesh/__init__.py
arna_3d/threeDrecon/mesh/conversion.py
arna_3d/threeDrecon/mesh/extraction.py
arna_3d/threeDrecon/mesh/reconstruction.py
arna_3d/threeDrecon/mesh/smoothing.py
arna_3d/threeDrecon/mesh/splitting.py
arna_3d/threeDrecon/mesh/transform.py
arna_3d/threeDrecon/segmentation/__init__.py
arna_3d/threeDrecon/segmentation/morphology.py
arna_3d/threeDrecon/segmentation/preprocessing.py
arna_3d/threeDrecon/vessel/__init__.py
arna_3d/threeDrecon/vessel/analysis.py
arna_3d/threeDrecon/vessel/branch.py
arna_3d/threeDrecon/vessel/interpolation.py
```

총 29개 Python 파일 + 2개 JSON 파일 생성됨.
