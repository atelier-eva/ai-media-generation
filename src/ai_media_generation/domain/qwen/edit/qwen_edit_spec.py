from dataclasses import dataclass
from pathlib import Path


@dataclass
class QwenEditSpec:
    id: str
    image: Path
    prompt: str
    negative: str = ""
