from dataclasses import dataclass


@dataclass
class SceneBackground:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
