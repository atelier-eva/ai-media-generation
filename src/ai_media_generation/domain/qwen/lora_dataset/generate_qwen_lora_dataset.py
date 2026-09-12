from ai_media_generation.domain.qwen.lora_dataset.camera import Camera
from ai_media_generation.domain.qwen.lora_dataset.expression import Expression
from ai_media_generation.domain.qwen.lora_dataset.generate_qwen_lora_dataset_output import (
    GenerateQwenLoraDatasetOutput,
    QwenLoraDatasetRow,
)
from ai_media_generation.domain.qwen.lora_dataset.generation import Generation
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.domain.qwen.lora_dataset.pose import Pose
from ai_media_generation.domain.qwen.lora_dataset.shoot import Shoot
from ai_media_generation.domain.qwen.lora_dataset.subject import Subject


class GenerateQwenLoraDataset:
    def execute(self, shoot: Shoot) -> GenerateQwenLoraDatasetOutput:
        rows: list[QwenLoraDatasetRow] = []
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
        return GenerateQwenLoraDatasetOutput(tuple(rows))

    def _caption_prompt(
        self,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedPrompt | None,
        background: NamedPrompt | None,
        lighting: NamedPrompt | None,
    ) -> str:
        caption = self._paragraph(
            subject.name,
            camera.angle.prompt,
            camera.distance.prompt,
            expression.prompt if expression else "",
            pose.prompt if pose else "",
            art_style.prompt if art_style else "",
            background.prompt if background else "",
            lighting.prompt if lighting else "",
        )
        if not caption:
            raise ValueError(
                "caption_prompt is empty for "
                f"{subject.name} / {camera.angle.name} / {camera.distance.name}."
            )
        return caption

    def _edit_prompt(
        self,
        generation: Generation,
        camera: Camera,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedPrompt | None,
        background: NamedPrompt | None,
        lighting: NamedPrompt | None,
    ) -> str:
        prompt = self._sentences(
            generation.prompt,
            self._paragraph(
                camera.angle.prompt,
                camera.distance.prompt,
                expression.prompt if expression else "",
                pose.prompt if pose else "",
                art_style.prompt if art_style else "",
                background.prompt if background else "",
                lighting.prompt if lighting else "",
            ),
        )
        if not prompt:
            raise ValueError(
                "edit_prompt is empty for "
                f"{camera.angle.name} / {camera.distance.name}."
            )
        return prompt

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

    def _optional(
        self, patterns: tuple[NamedPrompt, ...]
    ) -> tuple[NamedPrompt | None, ...]:
        if patterns:
            return patterns
        return (None,)

    def _paragraph(self, *parts: str) -> str:
        return " ".join(slot for slot in (self._slot(part) for part in parts) if slot)

    def _sentences(self, *parts: str) -> str:
        return " ".join(
            sentence
            for sentence in (self._sentence(part) for part in parts)
            if sentence
        )

    def _sentence(self, part: str) -> str:
        text = self._slot(part)
        if not text:
            return ""
        if text[0].islower():
            text = text[0].upper() + text[1:]
        return f"{text}."

    def _slot(self, part: str) -> str:
        return " ".join(part.split()).rstrip(".,;:")

    def _to_row(
        self,
        shoot: Shoot,
        camera: Camera,
        subject: Subject,
        expression: Expression | None,
        pose: Pose | None,
        art_style: NamedPrompt | None,
        background: NamedPrompt | None,
        lighting: NamedPrompt | None,
    ) -> QwenLoraDatasetRow:
        return QwenLoraDatasetRow(
            subject_name=subject.name,
            image=subject.image,
            angle_name=camera.angle.name,
            distance_name=camera.distance.name,
            expression_name=expression.name if expression else None,
            pose_name=pose.name if pose else None,
            art_style_name=art_style.name if art_style else None,
            background_name=background.name if background else None,
            lighting_name=lighting.name if lighting else None,
            edit_prompt=self._edit_prompt(
                shoot.generation,
                camera,
                expression,
                pose,
                art_style,
                background,
                lighting,
            ),
            caption_prompt=self._caption_prompt(
                camera,
                subject,
                expression,
                pose,
                art_style,
                background,
                lighting,
            ),
            negative_prompt=shoot.generation.negative.strip(),
        )
