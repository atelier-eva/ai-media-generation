from ai_media_generation.domain.animagine.lora_dataset.art_style.art_style import ArtStyle
from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera
from ai_media_generation.domain.animagine.lora_dataset.expression.expression import Expression
from ai_media_generation.domain.animagine.lora_dataset.pose.pose import Pose
from ai_media_generation.domain.animagine.lora_dataset.scene.background.scene_background import SceneBackground
from ai_media_generation.domain.animagine.lora_dataset.scene.lighting.scene_lighting import SceneLighting
from ai_media_generation.domain.animagine.lora_dataset.shoot.generate_shoot_patterns_output import (
    GenerateShootPatternsOutput,
)
from ai_media_generation.domain.animagine.lora_dataset.shoot.shoot import Shoot
from ai_media_generation.domain.animagine.lora_dataset.subject.feature.subject_feature import SubjectFeature
from ai_media_generation.domain.animagine.lora_dataset.subject.subject import Subject
from ai_media_generation.repository.animagine.lora_dataset.shoot_repository import ShootRepository


class GenerateShootPatterns:
    def execute(self) -> tuple[GenerateShootPatternsOutput, ...]:
        shoot = ShootRepository().find()
        patterns: list[GenerateShootPatternsOutput] = []
        for subject in shoot.subjects:
            for camera in shoot.cameras:
                for expression in self._expand_expressions(shoot, camera):
                    for pose in self._expand_poses(shoot, camera):
                        for art_style in self._expand_art_styles(shoot):
                            for background in self._expand_backgrounds(shoot):
                                for lighting in self._expand_lightings(shoot):
                                    patterns.append(
                                        self._to_output(
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
        return tuple(patterns)

    def _caption_prompt(
        self,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: ArtStyle | None,
        background: SceneBackground | None,
        lighting: SceneLighting | None,
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

    def _expand_art_styles(self, shoot: Shoot) -> tuple[ArtStyle | None, ...]:
        if shoot.art_styles:
            return shoot.art_styles
        return (None,)

    def _expand_backgrounds(
        self, shoot: Shoot
    ) -> tuple[SceneBackground | None, ...]:
        if shoot.backgrounds:
            return shoot.backgrounds
        return (None,)

    def _expand_expressions(
        self, shoot: Shoot, camera: Camera
    ) -> tuple[Expression | None, ...]:
        if shoot.expressions.includes(camera):
            return shoot.expressions.patterns
        return (None,)

    def _expand_lightings(self, shoot: Shoot) -> tuple[SceneLighting | None, ...]:
        if shoot.lightings:
            return shoot.lightings
        return (None,)

    def _expand_poses(self, shoot: Shoot, camera: Camera) -> tuple[Pose | None, ...]:
        if shoot.poses.includes(camera):
            return shoot.poses.patterns
        return (None,)

    def _feature_names(
        self, features: tuple[SubjectFeature, ...], *, weighted: bool = False
    ) -> tuple[str, ...]:
        return tuple(
            self._weighted_name(feature) if weighted else feature.name
            for feature in features
        )

    def _join(self, *groups: tuple[str, ...]) -> str:
        return ", ".join(name for group in groups for name in group if name)

    def _negative_prompt(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: ArtStyle | None,
        background: SceneBackground | None,
        lighting: SceneLighting | None,
    ) -> str:
        return self._join(
            shoot.rating.negative_features if shoot.rating else (),
            art_style.negative_features if art_style else (),
            shoot.quality.negative_features if shoot.quality else (),
            background.negative_features if background else (),
            lighting.negative_features if lighting else (),
            self._feature_names(subject.negative_features(camera)),
            camera.angle.negative_features,
            camera.distance.negative_features,
            expression.negative_features if expression else (),
            pose.negative_features if pose else (),
        )

    def _positive_prompt(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: ArtStyle | None,
        background: SceneBackground | None,
        lighting: SceneLighting | None,
    ) -> str:
        return self._join(
            subject.kinds,
            shoot.rating.positive_features if shoot.rating else (),
            art_style.positive_features if art_style else (),
            self._feature_names(subject.positive_features(camera), weighted=True),
            camera.angle.positive_features,
            camera.distance.positive_features,
            expression.positive_features if expression else (),
            pose.positive_features if pose else (),
            background.positive_features if background else (),
            lighting.positive_features if lighting else (),
            shoot.quality.positive_features if shoot.quality else (),
        )

    def _to_output(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: ArtStyle | None,
        background: SceneBackground | None,
        lighting: SceneLighting | None,
    ) -> GenerateShootPatternsOutput:
        return GenerateShootPatternsOutput(
            subject_name=subject.name,
            angle_name=camera.angle.name,
            distance_name=camera.distance.name,
            expression_name=expression.name if expression else None,
            pose_name=pose.name if pose else None,
            art_style_name=art_style.name if art_style else None,
            background_name=background.name if background else None,
            lighting_name=lighting.name if lighting else None,
            width=camera.frame.width,
            height=camera.frame.height,
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

    def _weighted_name(self, feature: SubjectFeature) -> str:
        if feature.weight is None:
            return feature.name
        return f"({feature.name}:{format(feature.weight, 'g')})"
