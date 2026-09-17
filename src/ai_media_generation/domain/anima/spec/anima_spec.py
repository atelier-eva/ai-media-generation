from dataclasses import dataclass


@dataclass
class AnimaSpec:
    id: str
    width: int
    height: int
    prompt: str
    negative: str = ""
