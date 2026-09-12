from ai_media_generation.domain.qwen.spec.get_qwen_specs_output import (
    GetQwenSpecsOutput,
)
from ai_media_generation.repository.qwen.spec_repository import QwenSpecRepository


class GetQwenSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetQwenSpecsOutput:
        return GetQwenSpecsOutput(QwenSpecRepository().get(ids))
