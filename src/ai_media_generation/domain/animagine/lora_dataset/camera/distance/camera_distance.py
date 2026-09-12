from dataclasses import dataclass

from ai_media_generation.domain.animagine.lora_dataset.camera.frame.camera_frame import CameraFrame


@dataclass
class CameraDistance:
    name: str
    frame: CameraFrame
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()
