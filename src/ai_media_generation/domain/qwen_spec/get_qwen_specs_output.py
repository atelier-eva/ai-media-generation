from dataclasses import dataclass

from ai_media_generation.domain.qwen_spec.qwen_spec import QwenSpec


@dataclass
class QwenSpecDto:
    id: str
    width: int
    height: int
    prompt: str
    negative: str


class GetQwenSpecsOutput:
    def __init__(self, specs: tuple[QwenSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: QwenSpec) -> QwenSpecDto:
        return QwenSpecDto(
            id=spec.id,
            width=spec.width,
            height=spec.height,
            prompt=spec.prompt,
            negative=spec.negative,
        )
