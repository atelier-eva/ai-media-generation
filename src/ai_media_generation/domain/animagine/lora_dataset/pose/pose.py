from dataclasses import dataclass


@dataclass
class Pose:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
