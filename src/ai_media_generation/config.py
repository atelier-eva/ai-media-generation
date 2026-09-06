from os import getenv
from pathlib import Path


class Config:
    LORA_TRAINING_DIRECTORY = "lora-training"
    LORA_TRAINING_GENERATIONS_JSONL = "lora-training-generations.jsonl"
    PROMPT_DIRECTORY = "prompt"
    MUSIC_DIRECTORY = "music"
    KAGEE_DIRECTORY = "kagee"
    IMAGE_OUTPUT_DIRECTORY = "image-output"
    MUSIC_OUTPUT_DIRECTORY = "music-output"
    KAGEE_OUTPUT_DIRECTORY = "kagee-output"
    LORA_DATASET_DIRECTORY = "lora-dataset"

    def __init__(self) -> None:
        self.comfy_ui_url = _text("COMFY_UI_URL").rstrip("/")
        self.comfy_ui_ckpt_name = _text("COMFY_UI_CKPT_NAME")
        self.comfy_ui_filename_prefix = _text("COMFY_UI_FILENAME_PREFIX")
        ace_step_url = _optional_text("ACE_STEP_URL")
        self.ace_step_url = ace_step_url.rstrip("/") if ace_step_url else None
        self.ace_step_api_key = _optional_text("ACE_STEP_API_KEY")
        self.ace_step_filename_prefix = (
            _optional_text("ACE_STEP_FILENAME_PREFIX") or "music"
        )
        kagee_timeout = _optional_int("KAGEE_TIMEOUT_SECONDS")
        if kagee_timeout is not None and kagee_timeout <= 0:
            raise ValueError("KAGEE_TIMEOUT_SECONDS must be positive.")
        self.kagee_timeout_seconds = 1800 if kagee_timeout is None else kagee_timeout

    @property
    def image_output_directory(self) -> Path:
        return _directory("IMAGE_OUTPUT_DIRECTORY", self.IMAGE_OUTPUT_DIRECTORY)

    @property
    def music_output_directory(self) -> Path:
        return _directory("MUSIC_OUTPUT_DIRECTORY", self.MUSIC_OUTPUT_DIRECTORY)

    @property
    def lora_dataset_directory(self) -> Path:
        return _directory("LORA_DATASET_DIRECTORY", self.LORA_DATASET_DIRECTORY)

    @property
    def lora_training_generations_jsonl(self) -> Path:
        return self.lora_dataset_directory / self.LORA_TRAINING_GENERATIONS_JSONL

    @property
    def lora_training_directory(self) -> Path:
        return _directory("LORA_TRAINING_DIRECTORY", self.LORA_TRAINING_DIRECTORY)

    @property
    def prompt_directory(self) -> Path:
        return _directory("PROMPT_DIRECTORY", self.PROMPT_DIRECTORY)

    @property
    def music_directory(self) -> Path:
        return _directory("MUSIC_DIRECTORY", self.MUSIC_DIRECTORY)

    @property
    def kagee_directory(self) -> Path:
        return _directory("KAGEE_DIRECTORY", self.KAGEE_DIRECTORY)

    @property
    def kagee_output_directory(self) -> Path:
        return _directory("KAGEE_OUTPUT_DIRECTORY", self.KAGEE_OUTPUT_DIRECTORY)

    @property
    def art_style_json(self) -> Path:
        return self.lora_training_directory / "art-style.json"

    @property
    def camera_json(self) -> Path:
        return self.lora_training_directory / "camera.json"

    @property
    def characters_directory(self) -> Path:
        return self.lora_training_directory / "characters"

    @property
    def expression_json(self) -> Path:
        return self.lora_training_directory / "expression.json"

    @property
    def generation_json(self) -> Path:
        return self.lora_training_directory / "generation.json"

    @property
    def pose_json(self) -> Path:
        return self.lora_training_directory / "pose.json"

    @property
    def scene_json(self) -> Path:
        return self.lora_training_directory / "scene.json"


def _directory(name: str, default: str) -> Path:
    specified = _optional_directory(name)
    if specified is not None:
        return specified
    path = Path(default).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    return path


def _optional_directory(name: str) -> Path | None:
    text = (getenv(name) or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    return path


def _optional_int(name: str) -> int | None:
    text = _optional_text(name)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer: {text}") from error


def _optional_text(name: str) -> str | None:
    value = (getenv(name) or "").strip()
    if not value:
        return None
    return value


def _text(name: str) -> str:
    value = (getenv(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is not set.")
    return value
