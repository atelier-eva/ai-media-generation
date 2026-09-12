from dataclasses import dataclass

from ai_media_generation.domain.animagine.spec.prompt import Prompt


@dataclass
class ImageSpec:
    id: str
    lora_id: str | None
    width: int
    height: int
    prompt: Prompt
    model_strength: float = 1.0
    text_encoder_strength: float = 1.0
    pose_id: str | None = None
