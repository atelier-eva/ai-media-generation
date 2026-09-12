from dataclasses import dataclass


@dataclass
class Generation:
    prompt: str
    negative: str = ""
