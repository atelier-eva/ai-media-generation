from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.camera.angle.camera_angle import CameraAngle
from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera
from ai_media_generation.domain.animagine.lora_dataset.camera.distance.camera_distance import CameraDistance
from ai_media_generation.domain.animagine.lora_dataset.camera.frame.camera_frame import CameraFrame
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class CameraRepository:
    def find(self) -> tuple[Camera, ...]:
        camera = read_json(Config().animagine_lora_training_camera_json)
        angles = tuple(self._to_angle(item) for item in camera.get("angle") or [])
        distances = tuple(
            self._to_distance(item) for item in camera.get("distance") or []
        )
        return tuple(
            Camera(angle=angle, distance=distance)
            for angle in angles
            for distance in distances
        )

    def _to_angle(self, data: dict[str, Any]) -> CameraAngle:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return CameraAngle(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )

    def _to_distance(self, data: dict[str, Any]) -> CameraDistance:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        size = data["image_size"]
        return CameraDistance(
            name=data["name"].strip(),
            frame=CameraFrame(width=size["width"], height=size["height"]),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
