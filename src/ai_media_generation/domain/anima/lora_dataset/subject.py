from dataclasses import dataclass
from enum import Enum

from ai_media_generation.domain.anima.lora_dataset.camera import Camera


class SubjectFeaturePolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class SubjectFeature:
    name: str
    polarity: SubjectFeaturePolarity
    skip_angles: tuple[str, ...] = ()
    skip_distances: tuple[str, ...] = ()

    def skips(self, camera: Camera) -> bool:
        if camera.angle.name in self.skip_angles:
            return True
        if camera.distance.name in self.skip_distances:
            return True
        return False


@dataclass
class Subject:
    name: str
    kinds: tuple[str, ...] = ()
    features: tuple[SubjectFeature, ...] = ()

    def filtered_features(self, camera: Camera) -> tuple[SubjectFeature, ...]:
        return tuple(
            feature for feature in self.features if not feature.skips(camera)
        )

    def negative_features(self, camera: Camera) -> tuple[SubjectFeature, ...]:
        return self._features(camera, SubjectFeaturePolarity.NEGATIVE)

    def positive_features(self, camera: Camera) -> tuple[SubjectFeature, ...]:
        features = self._features(camera, SubjectFeaturePolarity.POSITIVE)
        if not features:
            raise ValueError(
                f"Character '{self.name}' has no positive features for angle "
                f"'{camera.angle.name}' and distance '{camera.distance.name}'."
            )
        return features

    def _features(
        self, camera: Camera, polarity: SubjectFeaturePolarity
    ) -> tuple[SubjectFeature, ...]:
        return tuple(
            feature
            for feature in self.filtered_features(camera)
            if feature.polarity is polarity
        )
