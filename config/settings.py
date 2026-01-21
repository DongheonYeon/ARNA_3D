"""
설정 관리 모듈

파이프라인 설정과 스무딩 프리셋을 관리합니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class SmoothingConfig:
    """단일 파트의 스무딩 설정"""
    name: str
    smoothing_func: str | None = None
    smoothing_kwargs: dict[str, Any] = field(default_factory=dict)
    dilation_func: str | None = None
    dilation_kwargs: dict[str, Any] = field(default_factory=dict)
    simplification_func: str | None = None
    simplification_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmoothingConfig":
        """딕셔너리에서 SmoothingConfig 생성"""
        return cls(
            name=data["name"],
            smoothing_func=data.get("smoothing_func"),
            smoothing_kwargs=data.get("smoothing_kwargs") or {},
            dilation_func=data.get("dilation_func"),
            dilation_kwargs=data.get("dilation_kwargs") or {},
            simplification_func=data.get("simplification_func"),
            simplification_kwargs=data.get("simplification_kwargs") or {},
        )


@dataclass
class SmoothingPreset:
    """스무딩 프리셋 (여러 파트의 설정 모음)"""
    configs: list[SmoothingConfig] = field(default_factory=list)

    def get_config(self, part_name: str) -> SmoothingConfig | None:
        """파트 이름으로 설정 조회"""
        for config in self.configs:
            if config.name == part_name:
                return config
        return None

    def __iter__(self):
        return iter(self.configs)


def load_smoothing_preset(preset_path: Path | str) -> SmoothingPreset:
    """JSON 파일에서 스무딩 프리셋 로드"""
    preset_path = Path(preset_path)
    if not preset_path.exists():
        raise FileNotFoundError(f"프리셋 파일을 찾을 수 없습니다: {preset_path}")

    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs = [SmoothingConfig.from_dict(item) for item in data]
    return SmoothingPreset(configs=configs)


@dataclass
class PipelineSettings:
    """파이프라인 전체 설정"""
    # 경로 설정
    input_path: Path
    output_dir: Path | None = None

    # 디버그 설정
    debug: bool = False

    # 프리셋 경로 (기본값은 패키지 내 presets 폴더)
    stage1_preset_path: Path | None = None
    stage2_preset_path: Path | None = None

    def __post_init__(self):
        """경로 정규화 및 기본값 설정"""
        self.input_path = Path(self.input_path)

        # output_dir 기본값: input_path의 상위/3d/
        if self.output_dir is None:
            self.output_dir = self.input_path.parent.parent / "3d"
        else:
            self.output_dir = Path(self.output_dir)

        # 프리셋 경로 기본값
        presets_dir = Path(__file__).parent / "presets"
        if self.stage1_preset_path is None:
            self.stage1_preset_path = presets_dir / "stage1.json"
        if self.stage2_preset_path is None:
            self.stage2_preset_path = presets_dir / "stage2.json"

    @property
    def case_id(self) -> str | None:
        """경로에서 케이스 ID 추출 (case_XXXX 패턴)"""
        import re
        pattern = r'case_([a-zA-Z0-9\-_]+)'
        match = re.search(pattern, str(self.input_path))
        return match.group(1) if match else None

    @property
    def phase(self) -> str | None:
        """파일명에서 phase 추출 (segment_A 패턴)"""
        import re
        pattern = r'segment_([A-Z])'
        match = re.search(pattern, self.input_path.name)
        return match.group(1) if match else None

    def load_stage1_preset(self) -> SmoothingPreset:
        """1단계 스무딩 프리셋 로드"""
        return load_smoothing_preset(self.stage1_preset_path)

    def load_stage2_preset(self) -> SmoothingPreset:
        """2단계 스무딩 프리셋 로드"""
        return load_smoothing_preset(self.stage2_preset_path)
