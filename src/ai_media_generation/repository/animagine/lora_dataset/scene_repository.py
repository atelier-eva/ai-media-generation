from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.scene.background.scene_background import (
    SceneBackground,
)
from ai_media_generation.domain.animagine.lora_dataset.scene.lighting.scene_lighting import SceneLighting
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class SceneRepository:
    def find(self) -> tuple[tuple[SceneBackground, ...], tuple[SceneLighting, ...]]:
        data = read_json(Config().animagine_lora_training_scene_json)
        background = data.get("background") or {}
        lighting = data.get("lighting") or {}
        return (
            tuple(
                self._to_background(item) for item in background.get("patterns") or []
            ),
            tuple(self._to_lighting(item) for item in lighting.get("patterns") or []),
        )

    def _to_background(self, data: dict[str, Any]) -> SceneBackground:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return SceneBackground(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )

    def _to_lighting(self, data: dict[str, Any]) -> SceneLighting:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return SceneLighting(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
