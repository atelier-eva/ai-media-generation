from dataclasses import dataclass


@dataclass
class CameraAngle:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
