from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.kagee_spec.kagee_spec import KageeSpec
from ai_media_generation.domain.kagee_spec.prompt import TextPrompt
from ai_media_generation.repository.json_io import read_json


class KageeSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[KageeSpec, ...]:
        directory = self._kagee_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_kagee_spec(
                read_json(path, None), self._id_for(directory, path)
            )
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(f"Kagee JSON not found: {identifier}.json")
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid kagee id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No kagee JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _kagee_directory(self) -> Path:
        directory = Config().kagee_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"Kagee directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Kagee directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid kagee id: {identifier}")

    def _to_kagee_spec(self, data: dict[str, Any], identifier: str) -> KageeSpec:
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is empty or missing.")
        return KageeSpec(
            id=identifier,
            images=self._to_images(data.get("images")),
            prompt=TextPrompt(text=prompt),
            seed=self._to_seed(data.get("seed")),
        )

    def _to_images(self, value: Any) -> tuple[Path, ...]:
        if value is None:
            raise ValueError("images is empty or missing.")
        if not isinstance(value, list):
            raise ValueError("images must be an array.")
        paths: list[Path] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("images must contain paths as strings.")
            text = item.strip()
            if not text:
                raise ValueError("images contains an empty path.")
            path = Path(text).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Kagee input image not found: {path}")
            paths.append(path)
        if not paths:
            raise ValueError("images is empty or missing.")
        if len(paths) != 1:
            raise ValueError("multiple images are not implemented.")
        return tuple(paths)

    def _to_seed(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer.")
        return value
