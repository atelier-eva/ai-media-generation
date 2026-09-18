from dataclasses import dataclass
from pathlib import Path


@dataclass
class QwenLoraDatasetRow:
    subject_name: str
    images: tuple[Path, ...]
    angle_name: str
    distance_name: str
    expression_name: str | None
    pose_name: str | None
    art_style_name: str | None
    background_name: str | None
    lighting_name: str | None
    edit_prompt: str
    caption_prompt: str
    negative_prompt: str


class GenerateQwenLoraDatasetOutput:
    def __init__(self, rows: tuple[QwenLoraDatasetRow, ...]) -> None:
        self.rows = rows
