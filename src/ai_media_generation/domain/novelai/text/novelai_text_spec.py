from dataclasses import dataclass

DEFAULT_MODEL = "llama-3-erato-v1"
DEFAULT_MAX_LENGTH = 100


@dataclass
class NovelAiTextSpec:
    id: str
    input: str
    model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
