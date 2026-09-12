from dataclasses import dataclass, field

from ai_media_generation.domain.animagine.lora_dataset.art_style.art_style import ArtStyle
from ai_media_generation.domain.animagine.lora_dataset.camera.camera import Camera
from ai_media_generation.domain.animagine.lora_dataset.expression.expression_settings import ExpressionSettings
from ai_media_generation.domain.animagine.lora_dataset.pose.pose_settings import PoseSettings
from ai_media_generation.domain.animagine.lora_dataset.quality.quality import Quality
from ai_media_generation.domain.animagine.lora_dataset.rating.content_rating import ContentRating
from ai_media_generation.domain.animagine.lora_dataset.scene.background.scene_background import SceneBackground
from ai_media_generation.domain.animagine.lora_dataset.scene.lighting.scene_lighting import SceneLighting
from ai_media_generation.domain.animagine.lora_dataset.subject.subject import Subject


@dataclass
class Shoot:
    cameras: tuple[Camera, ...] = ()
    subjects: tuple[Subject, ...] = ()
    expressions: ExpressionSettings = field(default_factory=ExpressionSettings)
    poses: PoseSettings = field(default_factory=PoseSettings)
    backgrounds: tuple[SceneBackground, ...] = ()
    lightings: tuple[SceneLighting, ...] = ()
    art_styles: tuple[ArtStyle, ...] = ()
    quality: Quality | None = None
    rating: ContentRating | None = None
