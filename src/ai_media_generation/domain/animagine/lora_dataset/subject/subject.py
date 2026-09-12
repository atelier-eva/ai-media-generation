from dataclasses import dataclass

from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera
from ai_media_generation.domain.animagine.lora_dataset.subject.feature.subject_feature import (
    SubjectFeature,
    SubjectFeaturePolarity,
)


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
