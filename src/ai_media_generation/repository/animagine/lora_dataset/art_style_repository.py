from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.art_style.art_style import ArtStyle
from ai_media_generation.repository.json_io import (
    ANIMAGINE_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class ArtStyleRepository:
    def find(self) -> tuple[ArtStyle, ...]:
        art_style = read_json(
            Config().animagine_lora_training_art_style_json,
            ANIMAGINE_LORA_SCHEMAS["art-style.json"],
        )
        return tuple(
            self._to_art_style(item) for item in art_style.get("patterns") or []
        )

    def _to_art_style(self, data: dict[str, Any]) -> ArtStyle:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return ArtStyle(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
