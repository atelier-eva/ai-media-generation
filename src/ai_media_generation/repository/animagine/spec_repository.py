from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.spec.image_spec import ImageSpec
from ai_media_generation.domain.animagine.spec.prompt import Prompt
from ai_media_generation.repository.json_io import read_json, to_string_tuple


class ImageSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[ImageSpec, ...]:
        directory = self._prompt_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_image_spec(read_json(path), self._id_for(directory, path))
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(f"Prompt JSON not found: {identifier}.json")
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid prompt id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No prompt JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _prompt_directory(self) -> Path:
        directory = Config().animagine_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"Prompt directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Prompt directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid prompt id: {identifier}")

    def _to_image_spec(self, data: dict[str, Any], identifier: str) -> ImageSpec:
        lora = data.get("lora") or {}
        size = data["image_size"]
        return ImageSpec(
            id=identifier,
            lora_id=self._to_lora_id(data),
            width=size["width"],
            height=size["height"],
            prompt=Prompt(
                positive_features=to_string_tuple(data.get("positive")),
                negative_features=to_string_tuple(data.get("negative")),
            ),
            model_strength=self._to_strength(lora.get("strength_model")),
            text_encoder_strength=self._to_strength(lora.get("strength_clip")),
            pose_id=self._to_pose_id(data),
        )

    def _to_lora_id(self, data: dict[str, Any]) -> str | None:
        lora = data.get("lora")
        if not lora:
            return None
        identifier = str(lora["id"]).strip()
        if not identifier:
            raise ValueError("lora.id is empty or missing.")
        return identifier

    def _to_pose_id(self, data: dict[str, Any]) -> str | None:
        pose = data.get("pose")
        if not pose:
            return None
        identifier = str(pose["id"]).strip()
        if not identifier:
            raise ValueError("pose.id is empty or missing.")
        return identifier

    def _to_strength(self, value: Any, default: float = 1.0) -> float:
        if value is None:
            return default
        return float(value)
