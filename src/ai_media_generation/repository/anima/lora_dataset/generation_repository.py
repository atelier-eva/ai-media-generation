from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.generation import (
    FeatureSet,
    Generation,
)
from ai_media_generation.repository.json_io import (
    ANIMA_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class GenerationRepository:
    def find(self) -> Generation:
        data = read_json(
            Config().anima_lora_training_generation_json,
            ANIMA_LORA_SCHEMAS["generation.json"],
        )
        return Generation(
            quality=self._to_feature_set(data.get("quality")),
            rating=self._to_rating(data.get("rating")),
        )

    def _to_feature_set(self, data: Any) -> FeatureSet:
        if not data:
            return FeatureSet()
        return FeatureSet(
            positive_features=to_string_tuple(data.get("positive")),
            negative_features=to_string_tuple(data.get("negative")),
        )

    def _to_rating(self, data: Any) -> FeatureSet | None:
        if not data:
            return None
        return self._to_feature_set(data)
