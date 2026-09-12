from dataclasses import dataclass


@dataclass
class ContentRating:
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
