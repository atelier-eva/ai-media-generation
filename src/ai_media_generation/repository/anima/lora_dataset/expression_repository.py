from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.expression import Expression
from ai_media_generation.domain.anima.lora_dataset.expression_settings import (
    ExpressionSettings,
)
from ai_media_generation.repository.json_io import (
    ANIMA_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class ExpressionRepository:
    def find(self) -> ExpressionSettings:
        expression = read_json(
            Config().anima_lora_training_expression_json,
            ANIMA_LORA_SCHEMAS["expression.json"],
        )
        skip = expression.get("skip_camera") or {}
        return ExpressionSettings(
            patterns=tuple(
                self._to_expression(item) for item in expression.get("patterns") or []
            ),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_expression(self, data: dict[str, Any]) -> Expression:
        name = data["name"].strip()
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (name,)
        return Expression(
            name=name,
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
