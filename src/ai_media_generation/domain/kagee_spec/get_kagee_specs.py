from ai_media_generation.domain.kagee_spec.get_kagee_specs_output import (
    GetKageeSpecsOutput,
)
from ai_media_generation.repository.kagee_spec_repository import KageeSpecRepository


class GetKageeSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetKageeSpecsOutput:
        return GetKageeSpecsOutput(KageeSpecRepository().get(ids))
