from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.camera import Camera, CameraDistance
from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet
from ai_media_generation.repository.json_io import (
    ANIMA_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class CameraRepository:
    def find(self) -> tuple[Camera, ...]:
        camera = read_json(
            Config().anima_lora_training_camera_json,
            ANIMA_LORA_SCHEMAS["camera.json"],
        )
        angles = tuple(self._to_angle(item) for item in camera.get("angle") or [])
        distances = tuple(
            self._to_distance(item) for item in camera.get("distance") or []
        )
        return tuple(
            Camera(angle=angle, distance=distance)
            for angle in angles
            for distance in distances
        )

    def _to_angle(self, data: dict[str, Any]) -> NamedTagSet:
        name = data["name"].strip()
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (name,)
        return NamedTagSet(
            name=name,
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )

    def _to_distance(self, data: dict[str, Any]) -> CameraDistance:
        name = data["name"].strip()
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (name,)
        size = data["image_size"]
        return CameraDistance(
            name=name,
            width=size["width"],
            height=size["height"],
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
