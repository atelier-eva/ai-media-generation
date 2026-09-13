from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.repository.json_io import QWEN_LORA_SCHEMAS, read_json


class ArtStyleRepository:
    def find(self) -> tuple[NamedPrompt, ...]:
        art_style = read_json(
            Config().qwen_lora_training_art_style_json,
            QWEN_LORA_SCHEMAS["art-style.json"],
        )
        return tuple(
            self._to_named_prompt(item) for item in art_style.get("patterns") or []
        )

    def _to_named_prompt(self, data: dict[str, Any]) -> NamedPrompt:
        return NamedPrompt(
            name=data["name"].strip(),
            prompt=str(data["prompt"]).strip(),
        )
