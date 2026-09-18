from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet
from ai_media_generation.repository.json_io import (
    ANIMA_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class SceneRepository:
    def find(self) -> tuple[tuple[NamedTagSet, ...], tuple[NamedTagSet, ...]]:
        data = read_json(
            Config().anima_lora_training_scene_json,
            ANIMA_LORA_SCHEMAS["scene.json"],
        )
        background = data.get("background") or {}
        lighting = data.get("lighting") or {}
        return (
            tuple(
                self._to_named_tag_set(item)
                for item in background.get("patterns") or []
            ),
            tuple(
                self._to_named_tag_set(item) for item in lighting.get("patterns") or []
            ),
        )

    def _to_named_tag_set(self, data: dict[str, Any]) -> NamedTagSet:
        name = data["name"].strip()
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (name,)
        return NamedTagSet(
            name=name,
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
