from dataclasses import dataclass


@dataclass
class QwenSpec:
    id: str
    width: int
    height: int
    prompt: str
    negative: str = ""
