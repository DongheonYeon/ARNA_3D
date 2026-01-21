"""
ARNA-3D: 의료 영상 세그멘테이션을 고품질 3D GLB 모델로 변환하는 파이프라인

NIfTI 세그멘테이션 파일을 입력받아 5단계 처리를 거쳐 3D 메시를 생성합니다:
1. NIfTI 전처리 (혈관 분기 분할, Fat dilation)
2. Marching Cubes 메시 추출
3. 1단계 스무딩
4. Poisson 재구성
5. 2단계 스무딩
"""

__version__ = "1.0.0"
