from ai_media_generation.domain.qwen.edit.get_qwen_edit_specs_output import (
    GetQwenEditSpecsOutput,
)
from ai_media_generation.repository.qwen.edit_spec_repository import (
    QwenEditSpecRepository,
)


class GetQwenEditSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetQwenEditSpecsOutput:
        return GetQwenEditSpecsOutput(QwenEditSpecRepository().get(ids))
