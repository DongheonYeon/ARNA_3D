# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Settings

- 모든 설명과 응답은 한글로 제공
- 코드 주석은 한글로 작성

## Project Overview

ARNA-3D는 NIfTI 세그멘테이션 파일을 고품질 3D GLB 모델로 변환하는 의료 영상 처리 파이프라인입니다. 5단계 처리를 거칩니다: NIfTI 전처리, 메시 추출, 1단계 스무딩, Poisson 재구성, 2단계 스무딩.

## Common Commands

### Running the Pipeline

```bash
# 의존성 설치
pip install -r requirements.txt

# CLI로 실행 (프로젝트 루트에서)
python __main__.py <input_path> [--output-dir <dir>] [--debug]

# 예시
python __main__.py ./data/case_S004/mask/segment_A.nii.gz --debug
```

## Architecture

```
ARNA-3D/                     # 프로젝트 루트
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
├── domain/                  # 핵심 데이터 타입
│   └── types.py             # VolumeData, MeshCollection, ProcessingContext
│
├── file_io/                 # 입출력 (Python 내장 io와 충돌 방지)
│   ├── nifti.py             # NIfTI 읽기/쓰기
│   ├── mesh.py              # GLB/OBJ 읽기/쓰기
│   └── temp.py              # 임시 파일 관리 (context manager)
│
├── threeDrecon/             # 처리 로직
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
├── runner.py                # Pipeline 클래스, run_pipeline()
│
└── .legacy/                 # 레거시 코드 (참조용)
    ├── main.py
    └── threeDRecon/
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

스무딩 설정은 `config/presets/` 폴더의 JSON 파일로 관리됩니다:

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

- 프로젝트 루트(ARNA-3D/)에서 직접 실행
- 절대 임포트 사용 (예: `from config.constants import Label`)
- `file_io/` 폴더 사용 (Python 내장 `io` 모듈과 충돌 방지)
- 타입 힌팅 사용 (Python 3.10+ 문법)
- 단일 책임 원칙 준수 (함수당 하나의 역할)
- 임시 파일은 `TempFileManager` context manager로 관리

## Legacy Code (참조용)

기존 코드는 `.legacy/` 폴더에 보관:
- `main.py` → `runner.py`
- `threeDRecon/processNii.py` → `threeDrecon/vessel/`, `threeDrecon/segmentation/`
- `threeDRecon/combineGLB2.py` → `threeDrecon/mesh/extraction.py`
- `threeDRecon/processMesh.py` → `threeDrecon/mesh/smoothing.py`, `reconstruction.py`
