from ai_media_generation.domain.anima.spec.get_anima_specs_output import (
    GetAnimaSpecsOutput,
)
from ai_media_generation.repository.anima.spec_repository import AnimaSpecRepository


class GetAnimaSpecs:
    def execute(self, ids: tuple[str, ...] = ()) -> GetAnimaSpecsOutput:
        return GetAnimaSpecsOutput(AnimaSpecRepository().get(ids))
