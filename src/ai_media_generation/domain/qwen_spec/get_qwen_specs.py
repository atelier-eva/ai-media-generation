from ai_media_generation.domain.qwen_spec.get_qwen_specs_output import (
    GetQwenSpecsOutput,
)
from ai_media_generation.repository.qwen_spec_repository import QwenSpecRepository


class GetQwenSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetQwenSpecsOutput:
        return GetQwenSpecsOutput(QwenSpecRepository().get(ids))
