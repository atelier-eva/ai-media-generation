from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.camera import Camera
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.repository.json_io import QWEN_LORA_SCHEMAS, read_json


class CameraRepository:
    def find(self) -> tuple[Camera, ...]:
        camera = read_json(
            Config().qwen_lora_training_camera_json, QWEN_LORA_SCHEMAS["camera.json"]
        )
        angles = tuple(self._to_named_prompt(item) for item in camera.get("angle") or [])
        distances = tuple(
            self._to_named_prompt(item) for item in camera.get("distance") or []
        )
        return tuple(
            Camera(angle=angle, distance=distance)
            for angle in angles
            for distance in distances
        )

    def _to_named_prompt(self, data: dict[str, Any]) -> NamedPrompt:
        return NamedPrompt(
            name=data["name"].strip(),
            prompt=str(data["prompt"]).strip(),
        )
