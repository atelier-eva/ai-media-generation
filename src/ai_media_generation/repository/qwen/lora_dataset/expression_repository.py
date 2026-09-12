from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.expression import Expression
from ai_media_generation.domain.qwen.lora_dataset.expression_settings import (
    ExpressionSettings,
)
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class ExpressionRepository:
    def find(self) -> ExpressionSettings:
        expression = read_json(Config().qwen_lora_training_expression_json)
        skip = expression.get("skip_camera") or {}
        return ExpressionSettings(
            patterns=tuple(
                self._to_expression(item) for item in expression.get("patterns") or []
            ),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_expression(self, data: dict[str, Any]) -> Expression:
        return Expression(
            name=data["name"].strip(),
            prompt=str(data["prompt"]).strip(),
        )
