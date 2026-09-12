from dataclasses import dataclass

from ai_media_generation.domain.qwen.lora_dataset.camera import Camera
from ai_media_generation.domain.qwen.lora_dataset.pose import Pose


@dataclass
class PoseSettings:
    patterns: tuple[Pose, ...] = ()
    skip_angles: tuple[str, ...] = ()
    skip_distances: tuple[str, ...] = ()

    def includes(self, camera: Camera) -> bool:
        if camera.angle.name in self.skip_angles:
            return False
        if camera.distance.name in self.skip_distances:
            return False
        return bool(self.patterns)
