from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.spec.anima_spec import AnimaSpec
from ai_media_generation.repository.json_io import ANIMA_SPEC_SCHEMA, read_json


class AnimaSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[AnimaSpec, ...]:
        directory = self._anima_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_anima_spec(
                read_json(path, ANIMA_SPEC_SCHEMA), self._id_for(directory, path)
            )
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(f"Anima JSON not found: {identifier}.json")
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid anima id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No anima JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _anima_directory(self) -> Path:
        directory = Config().anima_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"Anima directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Anima directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid anima id: {identifier}")

    def _to_anima_spec(self, data: dict[str, Any], identifier: str) -> AnimaSpec:
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is empty or missing.")
        size = data["image_size"]
        return AnimaSpec(
            id=identifier,
            width=size["width"],
            height=size["height"],
            prompt=prompt,
            negative=str(data.get("negative") or "").strip(),
        )
