from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen_spec.qwen_spec import QwenSpec
from ai_media_generation.repository.json_io import read_json


class QwenSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[QwenSpec, ...]:
        directory = self._qwen_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_qwen_spec(read_json(path), self._id_for(directory, path))
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(f"Qwen JSON not found: {identifier}.json")
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid qwen id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No qwen JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _qwen_directory(self) -> Path:
        directory = Config().qwen_directory
        if not directory.exists():
            raise FileNotFoundError(f"Qwen directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Qwen directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid qwen id: {identifier}")

    def _to_qwen_spec(self, data: dict[str, Any], identifier: str) -> QwenSpec:
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is empty or missing.")
        size = data["image_size"]
        return QwenSpec(
            id=identifier,
            width=size["width"],
            height=size["height"],
            prompt=prompt,
            negative=str(data.get("negative") or "").strip(),
        )
