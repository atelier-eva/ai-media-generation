import json
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

SchemaResource = tuple[str, ...]

ANIMAGINE_PROMPT_SCHEMA: SchemaResource = ("animagine", "prompt.schema.json")
QWEN_SPEC_SCHEMA: SchemaResource = ("qwen", "qwen.schema.json")
QWEN_EDIT_SPEC_SCHEMA: SchemaResource = ("qwen", "edit", "qwen-edit.schema.json")
ANIMA_SPEC_SCHEMA: SchemaResource = ("anima", "anima.schema.json")
MUSIC_SCHEMA: SchemaResource = ("music.schema.json",)

ANIMAGINE_LORA_SCHEMAS: dict[str, SchemaResource] = {
    "art-style.json": ("animagine", "lora_dataset", "art-style.schema.json"),
    "camera.json": ("animagine", "lora_dataset", "camera.schema.json"),
    "characters": ("animagine", "lora_dataset", "characters.schema.json"),
    "expression.json": ("animagine", "lora_dataset", "expression.schema.json"),
    "generation.json": ("animagine", "lora_dataset", "generation.schema.json"),
    "pose.json": ("animagine", "lora_dataset", "pose.schema.json"),
    "scene.json": ("animagine", "lora_dataset", "scene.schema.json"),
}

QWEN_LORA_SCHEMAS: dict[str, SchemaResource] = {
    "art-style.json": ("qwen", "lora_dataset", "art-style.schema.json"),
    "camera.json": ("qwen", "lora_dataset", "camera.schema.json"),
    "characters": ("qwen", "lora_dataset", "characters.schema.json"),
    "expression.json": ("qwen", "lora_dataset", "expression.schema.json"),
    "generation.json": ("qwen", "lora_dataset", "generation.schema.json"),
    "pose.json": ("qwen", "lora_dataset", "pose.schema.json"),
    "scene.json": ("qwen", "lora_dataset", "scene.schema.json"),
}

ANIMA_LORA_SCHEMAS: dict[str, SchemaResource] = {
    "art-style.json": ("anima", "lora_dataset", "art-style.schema.json"),
    "camera.json": ("anima", "lora_dataset", "camera.schema.json"),
    "characters": ("anima", "lora_dataset", "characters.schema.json"),
    "expression.json": ("anima", "lora_dataset", "expression.schema.json"),
    "generation.json": ("anima", "lora_dataset", "generation.schema.json"),
    "pose.json": ("anima", "lora_dataset", "pose.schema.json"),
    "scene.json": ("anima", "lora_dataset", "scene.schema.json"),
}


def read_json(path: Path, schema: SchemaResource | None) -> dict[str, Any]:
    return _read_json(path.expanduser().resolve(), schema)


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
def _read_json(path: Path, schema: SchemaResource | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    loaded = _load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid JSON: {path}: JSON must be an object")
    if schema is None:
        return loaded
    validator = _validator(schema)
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


@cache
def _validator(resource: SchemaResource) -> Draft202012Validator:
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
