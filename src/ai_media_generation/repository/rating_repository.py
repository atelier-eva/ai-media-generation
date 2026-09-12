from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.rating.content_rating import ContentRating
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class RatingRepository:
    def find(self) -> ContentRating | None:
        return self._to_rating(
            read_json(Config().animagine_lora_training_generation_json).get("rating")
        )

    def _to_rating(self, data: Any) -> ContentRating | None:
        if not data:
            return None
        return ContentRating(
            positive_features=to_string_tuple(data.get("positive")),
            negative_features=to_string_tuple(data.get("negative")),
        )
