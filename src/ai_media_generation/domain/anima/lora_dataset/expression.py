from dataclasses import dataclass


@dataclass
class Expression:
    name: str
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
