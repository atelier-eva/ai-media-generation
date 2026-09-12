from dataclasses import dataclass


@dataclass
class GenerateShootPatternsOutput:
    subject_name: str
    angle_name: str
    distance_name: str
    expression_name: str | None
    pose_name: str | None
    art_style_name: str | None
    background_name: str | None
    lighting_name: str | None
    width: int
    height: int
    positive_prompt: str
    negative_prompt: str
    caption_prompt: str
