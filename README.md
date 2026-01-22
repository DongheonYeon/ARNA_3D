# ARNA-3D Medical Image Processing Pipeline

NIfTI 세그멘테이션 데이터를 3D GLB 모델로 변환하는 의료 영상 처리 파이프라인입니다.

## Installation

```bash
pip install -r requirements.txt
```

## Architecture

```
ARNA-3D/
├── core.py              # 백엔드 통합용 래퍼
├── pipeline.py          # 핵심 파이프라인 (Pipeline 클래스)
├── config/              # 설정 및 상수
│   ├── constants.py     # Label enum, 파라미터 상수
│   ├── settings.py      # PipelineSettings, SmoothingPreset
│   ├── logger.py        # SimpleLogger
│   └── presets/         # 스무딩 프리셋 JSON
├── domain/              # 핵심 데이터 타입
│   └── types.py         # VolumeData, MeshCollection
├── file_io/             # 파일 입출력
│   ├── nifti.py         # NIfTI 읽기/쓰기
│   └── mesh.py          # GLB/OBJ 읽기/쓰기
└── threeDrecon/         # 처리 로직
    ├── segmentation/    # 세그멘테이션 전처리
    ├── vessel/          # 혈관 분석
    └── mesh/            # 메시 처리
```

### Processing Pipeline

1. **NIfTI 로드**: 세그멘테이션 파일 로드
2. **전처리**: 혈관 분기 자동 분할, Fat dilation
3. **메시 추출**: Marching Cubes로 라벨별 메시 생성
4. **1단계 스무딩**: 구조별 Taubin/Laplacian 스무딩
5. **Poisson 재구성**: 혈관 메시 병합 및 재구성
6. **2단계 스무딩**: 마무리 스무딩
7. **GLB 저장**: 최종 3D 모델 출력

## Label for Anatomical Structures

| Label   | Structure       | ID  |
| ------- | --------------- | --- |
| Tumor   | Kidney Tumor    | 1   |
| Kidney  | Kidney          | 2   |
| Artery  | Arterial system | 3   |
| Vein    | Venous system   | 4   |
| Ureter  | Ureter          | 5   |
| Fat     | Surrounding fat | 6   |
| Renal_a | Renal artery    | 7   |
| Renal_v | Renal vein      | 8   |

## Configuration

스무딩 설정은 `config/presets/` 폴더의 JSON 파일로 관리됩니다:

- `stage1.json`: 1단계 스무딩 (주요 형태 정의)
- `stage2.json`: 2단계 스무딩 (마무리)

### 설정 예시

```json
{
  "name": "Tumor",
  "smoothing_func": "taubin",
  "smoothing_kwargs": {
    "n_iter": 100,
    "pass_band": 0.001,
    "feature_angle": 200.0,
    "boundary_smoothing": true
  },
  "dilation_func": "default",
  "dilation_kwargs": {
    "offset": -0.5
  }
}
```

## Data Structure

```
data/
├── case_XXXX/
│   ├── mask/segment_A.nii.gz    # 입력 세그멘테이션
│   └── 3d/obj_A.glb             # 출력 3D 모델
```

## Debug

디버그 모드 활성화 시 중간 결과가 저장됩니다:

```python
run_pipeline(input_path, output_path, debug=True)
```

저장되는 파일:

- `obj_{phase}_before_step1.glb`: 스무딩 전 메시
- `obj_{phase}_after_step1.glb`: 1단계 스무딩 후 메시

## Performance

일반적인 처리 시간 (하드웨어 및 해상도에 따라 상이):

- 총 처리 시간: ~40초 (Ryzen 5 7600 / RTX 4070 Ti Super / 32GB RAM / 1mm 이미지)

## Libraries

- [SimpleITK](https://simpleitk.org/)
- [Trimesh](https://trimsh.org/)
- [PyVista](https://pyvista.org/)
- [Open3D](http://www.open3d.org/)
- [VTK](https://vtk.org/)
