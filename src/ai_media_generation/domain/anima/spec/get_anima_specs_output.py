from dataclasses import dataclass

from ai_media_generation.domain.anima.spec.anima_spec import AnimaSpec


@dataclass
class AnimaSpecDto:
    id: str
    width: int
    height: int
    prompt: str
    negative: str


class GetAnimaSpecsOutput:
    def __init__(self, specs: tuple[AnimaSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: AnimaSpec) -> AnimaSpecDto:
        return AnimaSpecDto(
            id=spec.id,
            width=spec.width,
            height=spec.height,
            prompt=spec.prompt,
            negative=spec.negative,
        )
