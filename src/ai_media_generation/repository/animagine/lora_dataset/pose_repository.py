from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.pose.pose import Pose
from ai_media_generation.domain.animagine.lora_dataset.pose.pose_settings import PoseSettings
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class PoseRepository:
    def find(self) -> PoseSettings:
        pose = read_json(Config().animagine_lora_training_pose_json)
        skip = pose.get("skip_camera") or {}
        return PoseSettings(
            patterns=tuple(self._to_pose(item) for item in pose.get("patterns") or []),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_pose(self, data: dict[str, Any]) -> Pose:
        positive = to_string_tuple(data.get("positive"))
        if not positive:
            positive = (data["name"].strip(),)
        return Pose(
            name=data["name"].strip(),
            positive_features=positive,
            negative_features=to_string_tuple(data.get("negative")),
        )
