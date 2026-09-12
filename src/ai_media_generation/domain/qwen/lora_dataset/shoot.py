from dataclasses import dataclass, field

from ai_media_generation.domain.qwen.lora_dataset.camera import Camera
from ai_media_generation.domain.qwen.lora_dataset.expression_settings import (
    ExpressionSettings,
)
from ai_media_generation.domain.qwen.lora_dataset.generation import Generation
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.domain.qwen.lora_dataset.pose_settings import PoseSettings
from ai_media_generation.domain.qwen.lora_dataset.subject import Subject


@dataclass
class Shoot:
    generation: Generation
    cameras: tuple[Camera, ...] = ()
    subjects: tuple[Subject, ...] = ()
    expressions: ExpressionSettings = field(default_factory=ExpressionSettings)
    poses: PoseSettings = field(default_factory=PoseSettings)
    art_styles: tuple[NamedPrompt, ...] = ()
    backgrounds: tuple[NamedPrompt, ...] = ()
    lightings: tuple[NamedPrompt, ...] = ()
