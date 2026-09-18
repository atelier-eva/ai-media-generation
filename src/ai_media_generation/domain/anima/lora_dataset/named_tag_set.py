from dataclasses import dataclass


@dataclass
class NamedTagSet:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
