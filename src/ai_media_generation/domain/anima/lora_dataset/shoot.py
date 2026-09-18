from dataclasses import dataclass, field

from ai_media_generation.domain.anima.lora_dataset.camera import Camera
from ai_media_generation.domain.anima.lora_dataset.expression_settings import (
    ExpressionSettings,
)
from ai_media_generation.domain.anima.lora_dataset.generation import Generation
from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet
from ai_media_generation.domain.anima.lora_dataset.pose_settings import PoseSettings
from ai_media_generation.domain.anima.lora_dataset.subject import Subject


@dataclass
class Shoot:
    cameras: tuple[Camera, ...] = ()
    subjects: tuple[Subject, ...] = ()
    expressions: ExpressionSettings = field(default_factory=ExpressionSettings)
    poses: PoseSettings = field(default_factory=PoseSettings)
    backgrounds: tuple[NamedTagSet, ...] = ()
    lightings: tuple[NamedTagSet, ...] = ()
    art_styles: tuple[NamedTagSet, ...] = ()
    generation: Generation = field(default_factory=Generation)
