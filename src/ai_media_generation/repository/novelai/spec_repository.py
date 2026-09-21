from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.novelai.spec.novelai_spec import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLER,
    DEFAULT_SCALE,
    DEFAULT_STEPS,
    NovelAiSpec,
)
from ai_media_generation.repository.json_io import (
    NOVELAI_SPEC_SCHEMA,
    read_json,
    to_string_tuple,
)


class NovelAiSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[NovelAiSpec, ...]:
        directory = self._novelai_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_novelai_spec(
                read_json(path, NOVELAI_SPEC_SCHEMA), self._id_for(directory, path)
            )
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(f"NovelAI JSON not found: {identifier}.json")
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid novelai id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No novelai JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _novelai_directory(self) -> Path:
        directory = Config().novelai_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"NovelAI directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"NovelAI directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid novelai id: {identifier}")

    def _to_novelai_spec(self, data: dict[str, Any], identifier: str) -> NovelAiSpec:
        size = data["image_size"]
        model = str(data.get("model") or "").strip() or DEFAULT_MODEL
        sampler = str(data.get("sampler") or "").strip() or DEFAULT_SAMPLER
        return NovelAiSpec(
            id=identifier,
            width=size["width"],
            height=size["height"],
            positive=to_string_tuple(data.get("positive")),
            negative=to_string_tuple(data.get("negative")),
            model=model,
            sampler=sampler,
            steps=DEFAULT_STEPS if data.get("steps") is None else int(data["steps"]),
            scale=DEFAULT_SCALE if data.get("scale") is None else float(data["scale"]),
        )
