from dataclasses import dataclass

from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet


@dataclass
class CameraDistance:
    name: str
    width: int
    height: int
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()


@dataclass
class Camera:
    angle: NamedTagSet
    distance: CameraDistance

    @property
    def width(self) -> int:
        return self.distance.width

    @property
    def height(self) -> int:
        return self.distance.height
