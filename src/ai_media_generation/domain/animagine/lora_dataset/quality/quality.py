from dataclasses import dataclass


@dataclass
class Quality:
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
