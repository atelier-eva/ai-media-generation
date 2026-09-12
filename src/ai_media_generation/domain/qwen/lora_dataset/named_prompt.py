from dataclasses import dataclass


@dataclass
class NamedPrompt:
    name: str
    prompt: str
