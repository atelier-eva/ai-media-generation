from dataclasses import dataclass
from pathlib import Path

from ai_media_generation.domain.kagee_spec.kagee_spec import KageeSpec


@dataclass
class KageeSpecDto:
    id: str
    images: tuple[Path, ...]
    prompt: str
    seed: int | None


class GetKageeSpecsOutput:
    def __init__(self, specs: tuple[KageeSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: KageeSpec) -> KageeSpecDto:
        return KageeSpecDto(
            id=spec.id,
            images=spec.images,
            prompt=spec.prompt.text,
            seed=spec.seed,
        )
