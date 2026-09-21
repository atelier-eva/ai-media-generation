from dataclasses import dataclass

DEFAULT_MODEL = "nai-diffusion-5-full"
DEFAULT_SAMPLER = "k_euler_ancestral"
DEFAULT_STEPS = 28
DEFAULT_SCALE = 5.0
DEFAULT_USE_ORDER = True
_MAX_CHARACTERS_V5 = 22
_MAX_CHARACTERS_V4_5 = 6
_MAX_CHARACTERS_BY_MODEL = {
    "nai-diffusion-5-full": _MAX_CHARACTERS_V5,
    "nai-diffusion-5-curated": _MAX_CHARACTERS_V5,
    "nai-diffusion-4-5-full": _MAX_CHARACTERS_V4_5,
    "nai-diffusion-4-5-curated": _MAX_CHARACTERS_V4_5,
}


@dataclass
class NovelAiCharacter:
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        if (self.x is None) != (self.y is None):
            raise ValueError(
                "NovelAI character x and y must both be set or both omitted."
            )
        if self.x is not None and not 0 <= self.x <= 1:
            raise ValueError("NovelAI character x must be between 0 and 1.")
        if self.y is not None and not 0 <= self.y <= 1:
            raise ValueError("NovelAI character y must be between 0 and 1.")


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
    characters: tuple[NovelAiCharacter, ...] = ()
    use_order: bool = DEFAULT_USE_ORDER

    def __post_init__(self) -> None:
        limit = _MAX_CHARACTERS_BY_MODEL.get(self.model, _MAX_CHARACTERS_V5)
        count = len(self.characters)
        if count > limit:
            raise ValueError(
                f"NovelAI {self.id}: {self.model} allows at most {limit} "
                f"character prompts, got {count}."
            )
