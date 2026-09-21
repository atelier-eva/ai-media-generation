from ai_media_generation.domain.novelai.text.get_novelai_text_specs_output import (
    GetNovelAiTextSpecsOutput,
)
from ai_media_generation.repository.novelai.text_spec_repository import (
    NovelAiTextSpecRepository,
)


class GetNovelAiTextSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetNovelAiTextSpecsOutput:
        return GetNovelAiTextSpecsOutput(NovelAiTextSpecRepository().get(ids))
