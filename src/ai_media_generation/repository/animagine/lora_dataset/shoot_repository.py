from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.scene.background.scene_background import SceneBackground
from ai_media_generation.domain.animagine.lora_dataset.scene.lighting.scene_lighting import SceneLighting
from ai_media_generation.domain.animagine.lora_dataset.shoot.shoot import Shoot
from ai_media_generation.repository.animagine.lora_dataset.art_style_repository import ArtStyleRepository
from ai_media_generation.repository.animagine.lora_dataset.camera_repository import CameraRepository
from ai_media_generation.repository.animagine.lora_dataset.expression_repository import ExpressionRepository
from ai_media_generation.repository.animagine.lora_dataset.pose_repository import PoseRepository
from ai_media_generation.repository.animagine.lora_dataset.quality_repository import QualityRepository
from ai_media_generation.repository.animagine.lora_dataset.rating_repository import RatingRepository
from ai_media_generation.repository.animagine.lora_dataset.scene_repository import SceneRepository
from ai_media_generation.repository.animagine.lora_dataset.subject_repository import SubjectRepository


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
            quality=QualityRepository().find(),
            rating=RatingRepository().find(),
        )

    def _scene(self) -> tuple[tuple[SceneBackground, ...], tuple[SceneLighting, ...]]:
        if not Config().animagine_lora_training_scene_json.is_file():
            return ((), ())
        return SceneRepository().find()
