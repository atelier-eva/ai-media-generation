from dataclasses import dataclass


@dataclass
class SceneLighting:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
