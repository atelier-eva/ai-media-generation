from dataclasses import dataclass
from pathlib import Path

from ai_media_generation.domain.qwen.edit.qwen_edit_spec import QwenEditSpec


@dataclass
class QwenEditSpecDto:
    id: str
    image: Path
    prompt: str
    negative: str


class GetQwenEditSpecsOutput:
    def __init__(self, specs: tuple[QwenEditSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: QwenEditSpec) -> QwenEditSpecDto:
        return QwenEditSpecDto(
            id=spec.id,
            image=spec.image,
            prompt=spec.prompt,
            negative=spec.negative,
        )
