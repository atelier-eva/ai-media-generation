from ai_media_generation.domain.novelai.spec.get_novelai_specs_output import (
    GetNovelAiSpecsOutput,
)
from ai_media_generation.repository.novelai.spec_repository import NovelAiSpecRepository


class GetNovelAiSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetNovelAiSpecsOutput:
        return GetNovelAiSpecsOutput(NovelAiSpecRepository().get(ids))
