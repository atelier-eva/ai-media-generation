from os import getenv
from pathlib import Path


class Config:
    ANIMAGINE_SPEC_DIRECTORY = "animagine/spec"
    MUSIC_SPEC_DIRECTORY = "music"
    QWEN_SPEC_DIRECTORY = "qwen/spec"
    QWEN_EDIT_SPEC_DIRECTORY = "qwen/edit/spec"
    ANIMA_SPEC_DIRECTORY = "anima/spec"
    NOVELAI_SPEC_DIRECTORY = "novelai/spec"
    NOVELAI_TEXT_SPEC_DIRECTORY = "novelai/text/spec"
    NOVELAI_TEXT_LOREBOOK_DIRECTORY = "lorebook"
    NOVELAI_IMAGE_URL = "https://image.novelai.net"
    NOVELAI_TEXT_URL = "https://text.novelai.net"
    ANIMAGINE_OUTPUT_DIRECTORY = "animagine/output"
    MUSIC_OUTPUT_DIRECTORY = "music-output"
    QWEN_OUTPUT_DIRECTORY = "qwen/output"
    QWEN_EDIT_OUTPUT_DIRECTORY = "qwen/edit/output"
    ANIMA_OUTPUT_DIRECTORY = "anima/output"
    NOVELAI_OUTPUT_DIRECTORY = "novelai/output"
    NOVELAI_TEXT_OUTPUT_DIRECTORY = "novelai/text/output"
    ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY = "animagine/lora_dataset"
    ANIMAGINE_LORA_TRAINING_GENERATIONS_JSONL = (
        "animagine-lora-training-generations.jsonl"
    )
    ANIMAGINE_LORA_DATASET_DIRECTORY = "animagine/lora_output"
    QWEN_LORA_TRAINING_SPEC_DIRECTORY = "qwen/lora_dataset"
    QWEN_LORA_TRAINING_GENERATIONS_JSONL = "qwen-lora-training-generations.jsonl"
    QWEN_LORA_DATASET_DIRECTORY = "qwen/lora_output"
    ANIMA_LORA_TRAINING_SPEC_DIRECTORY = "anima/lora_dataset"
    ANIMA_LORA_TRAINING_GENERATIONS_JSONL = "anima-lora-training-generations.jsonl"
    ANIMA_LORA_DATASET_DIRECTORY = "anima/lora_output"

    def __init__(self) -> None:
        self.animagine_lora_filename_prefix = (
            _optional_text("ANIMAGINE_LORA_FILENAME_PREFIX") or "animagine-lora"
        )
        ace_step_url = _optional_text("ACE_STEP_URL")
        self.ace_step_url = ace_step_url.rstrip("/") if ace_step_url else None
        self.ace_step_api_key = _optional_text("ACE_STEP_API_KEY")
        self.ace_step_filename_prefix = (
            _optional_text("ACE_STEP_FILENAME_PREFIX") or "music"
        )
        qwen_timeout = _optional_int("QWEN_TIMEOUT_SECONDS")
        if qwen_timeout is not None and qwen_timeout <= 0:
            raise ValueError("QWEN_TIMEOUT_SECONDS must be positive.")
        self.qwen_timeout_seconds = 1800 if qwen_timeout is None else qwen_timeout
        qwen_edit_timeout = _optional_int("QWEN_EDIT_TIMEOUT_SECONDS")
        if qwen_edit_timeout is not None and qwen_edit_timeout <= 0:
            raise ValueError("QWEN_EDIT_TIMEOUT_SECONDS must be positive.")
        self.qwen_edit_timeout_seconds = (
            1800 if qwen_edit_timeout is None else qwen_edit_timeout
        )
        self.qwen_lora_filename_prefix = (
            _optional_text("QWEN_LORA_FILENAME_PREFIX") or "qwen-lora"
        )
        qwen_lora_timeout = _optional_int("QWEN_LORA_TIMEOUT_SECONDS")
        if qwen_lora_timeout is not None and qwen_lora_timeout <= 0:
            raise ValueError("QWEN_LORA_TIMEOUT_SECONDS must be positive.")
        self.qwen_lora_timeout_seconds = (
            1800 if qwen_lora_timeout is None else qwen_lora_timeout
        )
        anima_timeout = _optional_int("ANIMA_TIMEOUT_SECONDS")
        if anima_timeout is not None and anima_timeout <= 0:
            raise ValueError("ANIMA_TIMEOUT_SECONDS must be positive.")
        self.anima_timeout_seconds = 1800 if anima_timeout is None else anima_timeout
        self.anima_lora_filename_prefix = (
            _optional_text("ANIMA_LORA_FILENAME_PREFIX") or "anima-lora"
        )
        runpod_timeout = _optional_int("RUNPOD_TIMEOUT_SECONDS")
        if runpod_timeout is not None and runpod_timeout <= 0:
            raise ValueError("RUNPOD_TIMEOUT_SECONDS must be positive.")
        self.runpod_timeout_seconds = 600 if runpod_timeout is None else runpod_timeout
        self.novelai_api_token = _optional_text("NOVELAI_API_TOKEN")
        novelai_timeout = _optional_int("NOVELAI_TIMEOUT_SECONDS")
        if novelai_timeout is not None and novelai_timeout <= 0:
            raise ValueError("NOVELAI_TIMEOUT_SECONDS must be positive.")
        self.novelai_timeout_seconds = (
            120 if novelai_timeout is None else novelai_timeout
        )
        precision = _optional_text("QWEN_PRECISION")
        if precision is None:
            precision = "fp8"
        elif precision not in ("fp8", "bf16"):
            raise ValueError("QWEN_PRECISION must be fp8 or bf16.")
        self.qwen_precision = precision

    @property
    def comfy_ui_url(self) -> str:
        return _text("COMFY_UI_URL").rstrip("/")

    @property
    def comfy_ui_ckpt_name(self) -> str:
        return _text("ANIMAGINE_CKPT_NAME")

    @property
    def runpod_api_key(self) -> str:
        return _text("RUNPOD_API_KEY")

    @property
    def runpod_pod_id(self) -> str:
        return _text("RUNPOD_POD_ID")

    @property
    def runpod_ssh_identity(self) -> Path | None:
        text = _optional_text("RUNPOD_SSH_IDENTITY")
        if text is None:
            return None
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RUNPOD_SSH_IDENTITY is not a file: {path}")
        return path

    @property
    def runpod_comfyui_root(self) -> str:
        text = _optional_text("RUNPOD_COMFYUI_ROOT")
        root = "/workspace/runpod-slim/ComfyUI" if text is None else text.rstrip("/")
        if not root.startswith("/"):
            raise ValueError("RUNPOD_COMFYUI_ROOT must be an absolute path.")
        return root

    @property
    def animagine_output_directory(self) -> Path:
        return _directory("ANIMAGINE_OUTPUT_DIRECTORY", self.ANIMAGINE_OUTPUT_DIRECTORY)

    @property
    def music_output_directory(self) -> Path:
        return _directory("MUSIC_OUTPUT_DIRECTORY", self.MUSIC_OUTPUT_DIRECTORY)

    @property
    def animagine_lora_dataset_directory(self) -> Path:
        return _directory(
            "ANIMAGINE_LORA_DATASET_DIRECTORY",
            self.ANIMAGINE_LORA_DATASET_DIRECTORY,
        )

    @property
    def animagine_lora_training_generations_jsonl(self) -> Path:
        return (
            self.animagine_lora_dataset_directory
            / self.ANIMAGINE_LORA_TRAINING_GENERATIONS_JSONL
        )

    @property
    def animagine_lora_training_spec_directory(self) -> Path:
        return _directory(
            "ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY",
            self.ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY,
        )

    @property
    def animagine_spec_directory(self) -> Path:
        return _directory("ANIMAGINE_SPEC_DIRECTORY", self.ANIMAGINE_SPEC_DIRECTORY)

    @property
    def music_spec_directory(self) -> Path:
        return _directory("MUSIC_SPEC_DIRECTORY", self.MUSIC_SPEC_DIRECTORY)

    @property
    def qwen_spec_directory(self) -> Path:
        return _directory("QWEN_SPEC_DIRECTORY", self.QWEN_SPEC_DIRECTORY)

    @property
    def qwen_output_directory(self) -> Path:
        return _directory("QWEN_OUTPUT_DIRECTORY", self.QWEN_OUTPUT_DIRECTORY)

    @property
    def qwen_edit_spec_directory(self) -> Path:
        return _directory("QWEN_EDIT_SPEC_DIRECTORY", self.QWEN_EDIT_SPEC_DIRECTORY)

    @property
    def qwen_edit_output_directory(self) -> Path:
        return _directory(
            "QWEN_EDIT_OUTPUT_DIRECTORY", self.QWEN_EDIT_OUTPUT_DIRECTORY
        )

    @property
    def anima_spec_directory(self) -> Path:
        return _directory("ANIMA_SPEC_DIRECTORY", self.ANIMA_SPEC_DIRECTORY)

    @property
    def anima_output_directory(self) -> Path:
        return _directory("ANIMA_OUTPUT_DIRECTORY", self.ANIMA_OUTPUT_DIRECTORY)

    @property
    def novelai_spec_directory(self) -> Path:
        return _directory("NOVELAI_SPEC_DIRECTORY", self.NOVELAI_SPEC_DIRECTORY)

    @property
    def novelai_output_directory(self) -> Path:
        return _directory("NOVELAI_OUTPUT_DIRECTORY", self.NOVELAI_OUTPUT_DIRECTORY)

    @property
    def novelai_text_spec_directory(self) -> Path:
        return _directory(
            "NOVELAI_TEXT_SPEC_DIRECTORY", self.NOVELAI_TEXT_SPEC_DIRECTORY
        )

    def novelai_text_system_prompt(self, model: str) -> Path:
        return self.novelai_text_spec_directory.parent / f"system_prompt.{model}.txt"

    def novelai_text_memory(self, model: str) -> Path:
        return self.novelai_text_spec_directory.parent / f"memory.{model}.example"

    @property
    def novelai_text_lorebook_directory(self) -> Path:
        return (
            self.novelai_text_spec_directory.parent / self.NOVELAI_TEXT_LOREBOOK_DIRECTORY
        )

    @property
    def novelai_text_output_directory(self) -> Path:
        return _directory(
            "NOVELAI_TEXT_OUTPUT_DIRECTORY", self.NOVELAI_TEXT_OUTPUT_DIRECTORY
        )

    @property
    def anima_lora_training_spec_directory(self) -> Path:
        return _directory(
            "ANIMA_LORA_TRAINING_SPEC_DIRECTORY",
            self.ANIMA_LORA_TRAINING_SPEC_DIRECTORY,
        )

    @property
    def anima_lora_dataset_directory(self) -> Path:
        return _directory(
            "ANIMA_LORA_DATASET_DIRECTORY",
            self.ANIMA_LORA_DATASET_DIRECTORY,
        )

    @property
    def anima_lora_training_generations_jsonl(self) -> Path:
        return (
            self.anima_lora_dataset_directory
            / self.ANIMA_LORA_TRAINING_GENERATIONS_JSONL
        )

    @property
    def anima_lora_training_art_style_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "art-style.json"

    @property
    def anima_lora_training_camera_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "camera.json"

    @property
    def anima_lora_training_characters_directory(self) -> Path:
        return self.anima_lora_training_spec_directory / "characters"

    @property
    def anima_lora_training_expression_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "expression.json"

    @property
    def anima_lora_training_generation_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "generation.json"

    @property
    def anima_lora_training_pose_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "pose.json"

    @property
    def anima_lora_training_scene_json(self) -> Path:
        return self.anima_lora_training_spec_directory / "scene.json"

    @property
    def qwen_lora_training_spec_directory(self) -> Path:
        return _directory(
            "QWEN_LORA_TRAINING_SPEC_DIRECTORY",
            self.QWEN_LORA_TRAINING_SPEC_DIRECTORY,
        )

    @property
    def qwen_lora_dataset_directory(self) -> Path:
        return _directory(
            "QWEN_LORA_DATASET_DIRECTORY",
            self.QWEN_LORA_DATASET_DIRECTORY,
        )

    @property
    def qwen_lora_training_generations_jsonl(self) -> Path:
        return (
            self.qwen_lora_dataset_directory / self.QWEN_LORA_TRAINING_GENERATIONS_JSONL
        )

    @property
    def qwen_lora_training_art_style_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "art-style.json"

    @property
    def qwen_lora_training_camera_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "camera.json"

    @property
    def qwen_lora_training_characters_directory(self) -> Path:
        return self.qwen_lora_training_spec_directory / "characters"

    @property
    def qwen_lora_training_expression_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "expression.json"

    @property
    def qwen_lora_training_generation_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "generation.json"

    @property
    def qwen_lora_training_pose_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "pose.json"

    @property
    def qwen_lora_training_scene_json(self) -> Path:
        return self.qwen_lora_training_spec_directory / "scene.json"

    @property
    def animagine_lora_training_art_style_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "art-style.json"

    @property
    def animagine_lora_training_camera_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "camera.json"

    @property
    def animagine_lora_training_characters_directory(self) -> Path:
        return self.animagine_lora_training_spec_directory / "characters"

    @property
    def animagine_lora_training_expression_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "expression.json"

    @property
    def animagine_lora_training_generation_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "generation.json"

    @property
    def animagine_lora_training_pose_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "pose.json"

    @property
    def animagine_lora_training_scene_json(self) -> Path:
        return self.animagine_lora_training_spec_directory / "scene.json"


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
