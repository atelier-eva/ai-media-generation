from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.named_prompt import NamedPrompt
from ai_media_generation.domain.qwen.lora_dataset.shoot import Shoot
from ai_media_generation.repository.qwen.lora_dataset.art_style_repository import (
    ArtStyleRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.camera_repository import (
    CameraRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.expression_repository import (
    ExpressionRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.generation_repository import (
    GenerationRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.pose_repository import (
    PoseRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.scene_repository import (
    SceneRepository,
)
from ai_media_generation.repository.qwen.lora_dataset.subject_repository import (
    SubjectRepository,
)


class ShootRepository:
    def find(self) -> Shoot:
        backgrounds, lightings = self._scene()
        return Shoot(
            generation=GenerationRepository().find(),
            cameras=CameraRepository().find(),
            subjects=SubjectRepository().find(),
            expressions=ExpressionRepository().find(),
            poses=PoseRepository().find(),
            art_styles=ArtStyleRepository().find(),
            backgrounds=backgrounds,
            lightings=lightings,
        )

    def _scene(self) -> tuple[tuple[NamedPrompt, ...], tuple[NamedPrompt, ...]]:
        if not Config().qwen_lora_training_scene_json.is_file():
            return ((), ())
        return SceneRepository().find()
