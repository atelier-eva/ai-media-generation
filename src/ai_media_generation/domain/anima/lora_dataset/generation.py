from dataclasses import dataclass, field


@dataclass
class FeatureSet:
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()


@dataclass
class Generation:
    quality: FeatureSet = field(default_factory=FeatureSet)
    rating: FeatureSet | None = None
