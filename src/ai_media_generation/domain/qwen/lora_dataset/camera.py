from dataclasses import dataclass

from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt


@dataclass
class Camera:
    angle: NamedPrompt
    distance: NamedPrompt
