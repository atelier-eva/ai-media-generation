from dataclasses import dataclass

from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera
from ai_media_generation.domain.animagine.lora_dataset.expression.expression import Expression


@dataclass
class ExpressionSettings:
    patterns: tuple[Expression, ...] = ()
    skip_angles: tuple[str, ...] = ()
    skip_distances: tuple[str, ...] = ()

    def includes(self, camera: Camera) -> bool:
        if camera.angle.name in self.skip_angles:
            return False
        if camera.distance.name in self.skip_distances:
            return False
        return bool(self.patterns)
