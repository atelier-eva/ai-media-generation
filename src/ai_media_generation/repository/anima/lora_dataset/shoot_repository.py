from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.named_tag_set import NamedTagSet
from ai_media_generation.domain.anima.lora_dataset.shoot import Shoot
from ai_media_generation.repository.anima.lora_dataset.art_style_repository import (
    ArtStyleRepository,
)
from ai_media_generation.repository.anima.lora_dataset.camera_repository import (
    CameraRepository,
)
from ai_media_generation.repository.anima.lora_dataset.expression_repository import (
    ExpressionRepository,
)
from ai_media_generation.repository.anima.lora_dataset.generation_repository import (
    GenerationRepository,
)
from ai_media_generation.repository.anima.lora_dataset.pose_repository import (
    PoseRepository,
)
from ai_media_generation.repository.anima.lora_dataset.scene_repository import (
    SceneRepository,
)
from ai_media_generation.repository.anima.lora_dataset.subject_repository import (
    SubjectRepository,
)


class ShootRepository:
    def find(self) -> Shoot:
        backgrounds, lightings = self._scene()
        return Shoot(
            cameras=CameraRepository().find(),
            subjects=SubjectRepository().find(),
            expressions=ExpressionRepository().find(),
            poses=PoseRepository().find(),
            backgrounds=backgrounds,
            lightings=lightings,
            art_styles=ArtStyleRepository().find(),
            generation=GenerationRepository().find(),
        )

    def _scene(self) -> tuple[tuple[NamedTagSet, ...], tuple[NamedTagSet, ...]]:
        if not Config().anima_lora_training_scene_json.is_file():
            return ((), ())
        return SceneRepository().find()
