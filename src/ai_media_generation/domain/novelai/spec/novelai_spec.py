from dataclasses import dataclass

DEFAULT_MODEL = "nai-diffusion-5-full"
DEFAULT_SAMPLER = "k_euler_ancestral"
DEFAULT_STEPS = 28
DEFAULT_SCALE = 5.0


@dataclass
class NovelAiSpec:
    id: str
    width: int
    height: int
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    model: str = DEFAULT_MODEL
    sampler: str = DEFAULT_SAMPLER
    steps: int = DEFAULT_STEPS
    scale: float = DEFAULT_SCALE
