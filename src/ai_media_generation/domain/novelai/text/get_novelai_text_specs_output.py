from dataclasses import dataclass

from ai_media_generation.domain.novelai.text.novelai_text_spec import NovelAiTextSpec


@dataclass
class NovelAiTextSpecDto:
    id: str
    input: str
    model: str
    max_length: int
    output: str
    system_prompt: str = ""
    stop: tuple[str, ...] = ()


class GetNovelAiTextSpecsOutput:
    def __init__(self, specs: tuple[NovelAiTextSpec, ...]) -> None:
        self.dtos = tuple(self._to_dto(spec) for spec in specs)

    @staticmethod
    def _to_dto(spec: NovelAiTextSpec) -> NovelAiTextSpecDto:
        return NovelAiTextSpecDto(
            id=spec.id,
            input=spec.assembled_input(),
            model=spec.model,
            max_length=spec.max_length,
            output=spec.output,
            system_prompt=spec.system_prompt_text(),
            stop=spec.stop,
        )
