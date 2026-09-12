from dataclasses import dataclass


@dataclass
class ArtStyle:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
