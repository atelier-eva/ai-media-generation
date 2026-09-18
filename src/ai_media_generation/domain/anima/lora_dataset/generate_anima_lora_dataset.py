from ai_media_generation.domain.anima.lora_dataset.camera import Camera
from ai_media_generation.domain.anima.lora_dataset.expression import Expression
from ai_media_generation.domain.anima.lora_dataset.generate_anima_lora_dataset_output import (
    AnimaLoraDatasetRow,
    GenerateAnimaLoraDatasetOutput,
)
from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet
from ai_media_generation.domain.anima.lora_dataset.pose import Pose
from ai_media_generation.domain.anima.lora_dataset.shoot import Shoot
from ai_media_generation.domain.anima.lora_dataset.subject import Subject, SubjectFeature


class GenerateAnimaLoraDataset:
    def execute(self, shoot: Shoot) -> GenerateAnimaLoraDatasetOutput:
        rows: list[AnimaLoraDatasetRow] = []
        for subject in shoot.subjects:
            for camera in shoot.cameras:
                for expression in self._expand_expressions(shoot, camera):
                    for pose in self._expand_poses(shoot, camera):
                        for art_style in self._optional(shoot.art_styles):
                            for background in self._optional(shoot.backgrounds):
                                for lighting in self._optional(shoot.lightings):
                                    rows.append(
                                        self._to_row(
                                            shoot,
                                            camera,
                                            subject,
                                            expression,
                                            pose,
                                            art_style,
                                            background,
                                            lighting,
                                        )
                                    )
        return GenerateAnimaLoraDatasetOutput(tuple(rows))

    def _caption_prompt(
        self,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedTagSet | None,
        background: NamedTagSet | None,
        lighting: NamedTagSet | None,
    ) -> str:
        caption = self._join(
            (subject.name,),
            subject.kinds,
            self._feature_names(subject.positive_features(camera)),
            camera.angle.positive_features,
            camera.distance.positive_features,
            expression.positive_features if expression else (),
            pose.positive_features if pose else (),
            art_style.positive_features if art_style else (),
            background.positive_features if background else (),
            lighting.positive_features if lighting else (),
        )
        if not caption:
            raise ValueError(
                "caption_prompt is empty for "
                f"{subject.name} / {camera.angle.name} / {camera.distance.name}."
            )
        return caption

    def _expand_expressions(
        self, shoot: Shoot, camera: Camera
    ) -> tuple[Expression | None, ...]:
        if shoot.expressions.includes(camera):
            return shoot.expressions.patterns
        return (None,)

    def _expand_poses(self, shoot: Shoot, camera: Camera) -> tuple[Pose | None, ...]:
        if shoot.poses.includes(camera):
            return shoot.poses.patterns
        return (None,)

    def _feature_names(self, features: tuple[SubjectFeature, ...]) -> tuple[str, ...]:
        return tuple(feature.name for feature in features)

    def _join(self, *groups: tuple[str, ...]) -> str:
        return ", ".join(name for group in groups for name in group if name)

    def _negative_prompt(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedTagSet | None,
        background: NamedTagSet | None,
        lighting: NamedTagSet | None,
    ) -> str:
        rating = shoot.generation.rating
        return self._join(
            shoot.generation.quality.negative_features,
            rating.negative_features if rating else (),
            art_style.negative_features if art_style else (),
            background.negative_features if background else (),
            lighting.negative_features if lighting else (),
            self._feature_names(subject.negative_features(camera)),
            camera.angle.negative_features,
            camera.distance.negative_features,
            expression.negative_features if expression else (),
            pose.negative_features if pose else (),
        )

    def _optional(
        self, patterns: tuple[NamedTagSet, ...]
    ) -> tuple[NamedTagSet | None, ...]:
        if patterns:
            return patterns
        return (None,)

    def _positive_prompt(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedTagSet | None,
        background: NamedTagSet | None,
        lighting: NamedTagSet | None,
    ) -> str:
        rating = shoot.generation.rating
        return self._join(
            shoot.generation.quality.positive_features,
            rating.positive_features if rating else (),
            subject.kinds,
            self._feature_names(subject.positive_features(camera)),
            art_style.positive_features if art_style else (),
            camera.angle.positive_features,
            camera.distance.positive_features,
            expression.positive_features if expression else (),
            pose.positive_features if pose else (),
            background.positive_features if background else (),
            lighting.positive_features if lighting else (),
        )

    def _to_row(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedTagSet | None,
        background: NamedTagSet | None,
        lighting: NamedTagSet | None,
    ) -> AnimaLoraDatasetRow:
        return AnimaLoraDatasetRow(
            subject_name=subject.name,
            angle_name=camera.angle.name,
            distance_name=camera.distance.name,
            expression_name=expression.name if expression else None,
            pose_name=pose.name if pose else None,
            art_style_name=art_style.name if art_style else None,
            background_name=background.name if background else None,
            lighting_name=lighting.name if lighting else None,
            width=camera.width,
            height=camera.height,
            positive_prompt=self._positive_prompt(
                shoot,
                camera,
                subject,
                expression,
                pose,
                art_style,
                background,
                lighting,
            ),
            negative_prompt=self._negative_prompt(
                shoot,
                camera,
                subject,
                expression,
                pose,
                art_style,
                background,
                lighting,
            ),
            caption_prompt=self._caption_prompt(
                camera, subject, expression, pose, art_style, background, lighting
            ),
        )
