from dataclasses import dataclass

from ai_media_generation.domain.novelai.spec.novelai_spec import (
    NovelAiCharacter,
    NovelAiSpec,
)


@dataclass
class NovelAiCharacterDto:
    positive_prompt: str
    negative_prompt: str
    x: float | None
    y: float | None


@dataclass
class NovelAiSpecDto:
    id: str
    width: int
    height: int
    positive_prompt: str
    negative_prompt: str
    model: str
    sampler: str
    steps: int
    scale: float
    characters: tuple[NovelAiCharacterDto, ...]
    use_order: bool


class GetNovelAiSpecsOutput:
    def __init__(self, specs: tuple[NovelAiSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: NovelAiSpec) -> NovelAiSpecDto:
        return NovelAiSpecDto(
            id=spec.id,
            width=spec.width,
            height=spec.height,
            positive_prompt=_join(spec.positive),
            negative_prompt=_join(spec.negative),
            model=spec.model,
            sampler=spec.sampler,
            steps=spec.steps,
            scale=spec.scale,
            characters=tuple(
                _to_character_dto(character) for character in spec.characters
            ),
            use_order=spec.use_order,
        )


def _to_character_dto(character: NovelAiCharacter) -> NovelAiCharacterDto:
    return NovelAiCharacterDto(
        positive_prompt=_join(character.positive),
        negative_prompt=_join(character.negative),
        x=character.x,
        y=character.y,
    )


def _join(tags: tuple[str, ...]) -> str:
    return ", ".join(tag for tag in tags if tag)
