from ai_media_generation.config import Config
from ai_media_generation.domain.scene.background.scene_background import SceneBackground
from ai_media_generation.domain.scene.lighting.scene_lighting import SceneLighting
from ai_media_generation.domain.shoot.shoot import Shoot
from ai_media_generation.repository.art_style_repository import ArtStyleRepository
from ai_media_generation.repository.camera_repository import CameraRepository
from ai_media_generation.repository.expression_repository import ExpressionRepository
from ai_media_generation.repository.pose_repository import PoseRepository
from ai_media_generation.repository.quality_repository import QualityRepository
from ai_media_generation.repository.rating_repository import RatingRepository
from ai_media_generation.repository.scene_repository import SceneRepository
from ai_media_generation.repository.subject_repository import SubjectRepository


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
