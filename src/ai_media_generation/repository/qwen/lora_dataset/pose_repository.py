from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.pose import Pose
from ai_media_generation.domain.qwen.lora_dataset.pose_settings import PoseSettings
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class PoseRepository:
    def find(self) -> PoseSettings:
        pose = read_json(Config().qwen_lora_training_pose_json)
        skip = pose.get("skip_camera") or {}
        return PoseSettings(
            patterns=tuple(self._to_pose(item) for item in pose.get("patterns") or []),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_pose(self, data: dict[str, Any]) -> Pose:
        return Pose(
            name=data["name"].strip(),
            prompt=str(data["prompt"]).strip(),
        )
