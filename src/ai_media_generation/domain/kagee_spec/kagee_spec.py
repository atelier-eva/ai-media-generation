from dataclasses import dataclass
from pathlib import Path

from ai_media_generation.domain.kagee_spec.prompt import TextPrompt


@dataclass
class KageeSpec:
    id: str
    images: tuple[Path, ...]
    prompt: TextPrompt
    seed: int | None = None
