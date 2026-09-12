from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.expression.expression import Expression
from ai_media_generation.domain.animagine.lora_dataset.expression.expression_settings import ExpressionSettings
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class ExpressionRepository:
    def find(self) -> ExpressionSettings:
        expression = read_json(Config().animagine_lora_training_expression_json)
        skip = expression.get("skip_camera") or {}
        return ExpressionSettings(
            patterns=tuple(
                self._to_expression(item) for item in expression.get("patterns") or []
            ),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_expression(self, data: dict[str, Any]) -> Expression:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return Expression(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
