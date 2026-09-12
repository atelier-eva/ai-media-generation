from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.repository.json_io import read_json


class SceneRepository:
    def find(self) -> tuple[tuple[NamedPrompt, ...], tuple[NamedPrompt, ...]]:
        data = read_json(Config().qwen_lora_training_scene_json)
        background = data.get("background") or {}
        lighting = data.get("lighting") or {}
        return (
            tuple(
                self._to_named_prompt(item) for item in background.get("patterns") or []
            ),
            tuple(self._to_named_prompt(item) for item in lighting.get("patterns") or []),
        )

    def _to_named_prompt(self, data: dict[str, Any]) -> NamedPrompt:
        return NamedPrompt(
            name=data["name"].strip(),
            prompt=str(data["prompt"]).strip(),
        )
