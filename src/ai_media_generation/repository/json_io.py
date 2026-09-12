import json
from collections.abc import Callable
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from ai_media_generation.config import Config

_SCHEMA_RESOURCES = {
    "prompt": ("animagine", "prompt.schema.json"),
    "qwen": ("qwen", "qwen.schema.json"),
    "music": ("music.schema.json",),
}

_ANIMAGINE_LORA_SCHEMA_RESOURCES = {
    "art-style.json": ("animagine", "lora_dataset", "art-style.schema.json"),
    "camera.json": ("animagine", "lora_dataset", "camera.schema.json"),
    "characters": ("animagine", "lora_dataset", "characters.schema.json"),
    "expression.json": ("animagine", "lora_dataset", "expression.schema.json"),
    "generation.json": ("animagine", "lora_dataset", "generation.schema.json"),
    "pose.json": ("animagine", "lora_dataset", "pose.schema.json"),
    "scene.json": ("animagine", "lora_dataset", "scene.schema.json"),
}

_QWEN_LORA_SCHEMA_RESOURCES = {
    "art-style.json": ("qwen", "lora_dataset", "art-style.schema.json"),
    "camera.json": ("qwen", "lora_dataset", "camera.schema.json"),
    "characters": ("qwen", "lora_dataset", "characters.schema.json"),
    "expression.json": ("qwen", "lora_dataset", "expression.schema.json"),
    "generation.json": ("qwen", "lora_dataset", "generation.schema.json"),
    "pose.json": ("qwen", "lora_dataset", "pose.schema.json"),
    "scene.json": ("qwen", "lora_dataset", "scene.schema.json"),
}


def read_json(path: Path) -> dict[str, Any]:
    return _read_json(path.expanduser().resolve())


def read_resource_json(*relative: str) -> dict[str, Any]:
    return _read_resource_json(relative)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON: {path}: {error.msg} "
            f"(line {error.lineno} column {error.colno})"
        ) from error


@cache
def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    loaded = _load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid JSON: {path}: JSON must be an object")
    resource = _schema_resource(path)
    if resource is None:
        return loaded
    validator = _validator(resource)
    error = best_match(validator.iter_errors(loaded))
    if error is not None:
        raise ValueError(_format_validation_error(path, error)) from error
    return loaded


@cache
def _read_resource_json(relative: tuple[str, ...]) -> dict[str, Any]:
    resource = files("ai_media_generation.resources").joinpath(*relative)
    loaded = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON must be an object: {resource}")
    return loaded


def to_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(tag.strip() for tag in value if str(tag).strip())


def _schema_resource(path: Path) -> tuple[str, ...] | None:
    try:
        config = Config()
    except ValueError:
        return _schema_resource_by_path(path)
    qwen_lora_characters = _directory_or_none(
        lambda: config.qwen_lora_training_characters_directory
    )
    if qwen_lora_characters is not None and path.is_relative_to(qwen_lora_characters):
        return _QWEN_LORA_SCHEMA_RESOURCES["characters"]
    qwen_lora = _directory_or_none(lambda: config.qwen_lora_training_spec_directory)
    if qwen_lora is not None and path.is_relative_to(qwen_lora):
        return _QWEN_LORA_SCHEMA_RESOURCES.get(path.name)
    animagine_lora_characters = _directory_or_none(
        lambda: config.animagine_lora_training_characters_directory
    )
    if (
        animagine_lora_characters is not None
        and path.is_relative_to(animagine_lora_characters)
    ):
        return _ANIMAGINE_LORA_SCHEMA_RESOURCES["characters"]
    animagine_lora = _directory_or_none(
        lambda: config.animagine_lora_training_spec_directory
    )
    if animagine_lora is not None and path.is_relative_to(animagine_lora):
        return _ANIMAGINE_LORA_SCHEMA_RESOURCES.get(path.name)
    for key, directory in (
        ("prompt", lambda: config.animagine_spec_directory),
        ("qwen", lambda: config.qwen_spec_directory),
        ("music", lambda: config.music_spec_directory),
    ):
        root = _directory_or_none(directory)
        if root is not None and path.is_relative_to(root):
            return _SCHEMA_RESOURCES[key]
    return _schema_resource_by_path(path)


def _schema_resource_by_path(path: Path) -> tuple[str, ...] | None:
    parts = path.parts
    if "qwen" in parts and "lora_dataset" in parts:
        if "characters" in parts:
            return _QWEN_LORA_SCHEMA_RESOURCES["characters"]
        return _QWEN_LORA_SCHEMA_RESOURCES.get(path.name)
    if "animagine" in parts and "lora_dataset" in parts:
        if "characters" in parts:
            return _ANIMAGINE_LORA_SCHEMA_RESOURCES["characters"]
        return _ANIMAGINE_LORA_SCHEMA_RESOURCES.get(path.name)
    if "qwen" in parts and "spec" in parts:
        return _SCHEMA_RESOURCES["qwen"]
    if "animagine" in parts and "spec" in parts:
        return _SCHEMA_RESOURCES["prompt"]
    return _SCHEMA_RESOURCES.get(path.name)


def _directory_or_none(directory: Callable[[], Path]) -> Path | None:
    try:
        return directory()
    except (NotADirectoryError, ValueError):
        return None


@cache
def _validator(resource: tuple[str, ...]) -> Draft202012Validator:
    schema = json.loads(
        files("ai_media_generation.resources")
        .joinpath(*resource)
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _format_validation_error(path: Path, error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    suffix = f" at {location}" if location else ""
    return f"Invalid JSON: {path}: {error.message}{suffix}"
