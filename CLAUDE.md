# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Settings

- 모든 설명과 응답은 한글로 제공
- 코드 주석은 한글로 작성

## Project Overview

ARNA-3D는 NIfTI 세그멘테이션 파일을 고품질 3D GLB 모델로 변환하는 의료 영상 처리 파이프라인입니다. 5단계 처리를 거칩니다: NIfTI 전처리, 메시 추출, 1단계 스무딩, Poisson 재구성, 2단계 스무딩.

## Common Commands

### Running the Pipeline (새 패키지)

```bash
# 의존성 설치
pip install -r requirements.txt

# CLI로 실행
python -m arna_3d <input_path> [--output-dir <dir>] [--debug]

# 예시
python -m arna_3d ./data/case_S004/mask/segment_A.nii.gz --debug
```

### Running the Pipeline (기존 코드 - deprecated)

```bash
# 기존 main.py로 실행 (레거시)
python main.py
```

## Architecture (새 구조)

```
arna_3d/
├── __init__.py              # 패키지 초기화
├── __main__.py              # CLI 진입점
│
├── config/                  # 설정 및 상수
│   ├── constants.py         # Label enum, 파라미터 상수
│   ├── settings.py          # PipelineSettings, SmoothingPreset
│   └── presets/             # 스무딩 프리셋 JSON
│       ├── stage1.json
│       └── stage2.json
│
├── core/                    # 핵심 데이터 타입
│   ├── types.py             # VolumeData, MeshCollection, ProcessingContext
│   └── exceptions.py        # 커스텀 예외 클래스
│
├── io/                      # 입출력
│   ├── nifti.py             # NIfTI 읽기/쓰기
│   ├── mesh.py              # GLB/OBJ 읽기/쓰기
│   └── temp.py              # 임시 파일 관리 (context manager)
│
├── processing/              # 처리 로직
│   ├── segmentation/        # 세그멘테이션 전처리
│   │   ├── preprocessing.py # 라벨 전처리, Fat dilation
│   │   └── morphology.py    # 형태학적 연산
│   │
│   ├── vessel/              # 혈관 분석
│   │   ├── analysis.py      # 반경 분석, 범위 탐지
│   │   ├── interpolation.py # 원형/타원 보간
│   │   └── branch.py        # 분기 추출
│   │
│   └── mesh/                # 메시 처리
│       ├── extraction.py    # Marching Cubes 추출
│       ├── splitting.py     # L/R 분할, Tumor 검증
│       ├── smoothing.py     # Taubin/Laplacian 스무딩
│       ├── reconstruction.py # Poisson 재구성
│       ├── transform.py     # 회전, 중심 이동
│       └── conversion.py    # trimesh ↔ pyvista ↔ open3d
│
└── pipeline/                # 파이프라인 오케스트레이션
    └── runner.py            # Pipeline 클래스, run_pipeline()
```

### Core Processing Flow

1. **입력**: NIfTI 세그멘테이션 파일 (`segment_A.nii.gz`)
2. **전처리**: 혈관 분기 자동 분할, Fat dilation
3. **메시 추출**: Marching Cubes로 각 라벨별 메시 생성
4. **1단계 스무딩**: 구조별 Taubin/Laplacian 스무딩
5. **Poisson 재구성**: 혈관 메시 병합 및 재구성
6. **2단계 스무딩**: 마무리 스무딩
7. **출력**: GLB 파일 (`obj_A.glb`)

### Key Data Types

```python
# 볼륨 데이터
VolumeData(array, spacing, origin, direction)

# 메시 컬렉션
MeshCollection()  # dict-like, .add(), .get(), .to_scene()

# 파이프라인 설정
PipelineSettings(input_path, output_dir, debug)
```

### Anatomical Structure Labels

```python
class Label(IntEnum):
    TUMOR = 1
    KIDNEY = 2
    ARTERY = 3
    VEIN = 4
    URETER = 5
    FAT = 6
    RENAL_A = 7  # 동맥 분기
    RENAL_V = 8  # 정맥 분기
```

## Configuration

스무딩 설정은 `arna_3d/config/presets/` 폴더의 JSON 파일로 관리됩니다:

- `stage1.json`: 1단계 스무딩 (주요 형태 정의)
- `stage2.json`: 2단계 스무딩 (마무리)

각 구조별로 설정 가능한 항목:
- `smoothing_func`: "taubin" | "laplacian" | null
- `dilation_func`: "default" | null
- `simplification_func`: "decimate" | "decimate_pro" | null

## Data Structure

```
data/
├── case_XXXX/
│   ├── mask/segment_A.nii.gz    # 입력 세그멘테이션
│   └── 3d/obj_A.glb             # 출력 3D 모델
└── inference/                    # 추가 데이터셋
```

## Development Notes

- 새 코드는 `arna_3d/` 패키지 사용
- 기존 `threeDRecon/`과 `main.py`는 레거시 (향후 제거 예정)
- 타입 힌팅 사용 (Python 3.10+ 문법)
- 단일 책임 원칙 준수 (함수당 하나의 역할)
- 임시 파일은 `TempFileManager` context manager로 관리

## Legacy Code (deprecated)

기존 코드는 `threeDRecon/` 폴더에 있으며, 리팩토링 완료 후 제거 예정:
- `processNii.py` → `arna_3d/processing/vessel/`, `arna_3d/processing/segmentation/`
- `combineGLB2.py` → `arna_3d/processing/mesh/extraction.py`
- `processMesh.py` → `arna_3d/processing/mesh/smoothing.py`, `reconstruction.py`
- `main.py` → `arna_3d/pipeline/runner.py`
