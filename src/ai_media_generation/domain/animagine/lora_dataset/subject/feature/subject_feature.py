from dataclasses import dataclass
from enum import Enum

from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera


class SubjectFeaturePolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class SubjectFeature:
    name: str
    polarity: SubjectFeaturePolarity
    weight: float | None = None
    skip_angles: tuple[str, ...] = ()
    skip_distances: tuple[str, ...] = ()

    def skips(self, camera: Camera) -> bool:
        if camera.angle.name in self.skip_angles:
            return True
        if camera.distance.name in self.skip_distances:
            return True
        return False
