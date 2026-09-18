from dataclasses import dataclass
from pathlib import Path


@dataclass
class QwenEditSpec:
    id: str
    images: tuple[Path, ...]
    prompt: str
    negative: str = ""
