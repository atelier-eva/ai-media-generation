from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.novelai.text.novelai_text_spec import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    NovelAiTextSpec,
)
from ai_media_generation.repository.json_io import NOVELAI_TEXT_SPEC_SCHEMA, read_json


class NovelAiTextSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[NovelAiTextSpec, ...]:
        directory = self._text_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        return tuple(
            self._to_text_spec(
                read_json(path, NOVELAI_TEXT_SPEC_SCHEMA),
                self._id_for(directory, path),
            )
            for path in paths
        )

    def _paths_for(self, directory: Path, ids: tuple[str, ...]) -> tuple[Path, ...]:
        paths: list[Path] = []
        for identifier in ids:
            path = self._path_for(directory, identifier)
            if not path.is_file():
                raise FileNotFoundError(
                    f"NovelAI text JSON not found: {identifier}.json"
                )
            paths.append(path)
        return tuple(paths)

    def _path_for(self, directory: Path, identifier: str) -> Path:
        self._validate_id(identifier)
        path = (directory / f"{identifier}.json").expanduser().resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"Invalid novelai-text id: {identifier}")
        return path

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        paths = tuple(
            sorted(path for path in directory.rglob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No novelai-text JSON in {directory}")
        return paths

    def _id_for(self, directory: Path, path: Path) -> str:
        return path.resolve().relative_to(directory).with_suffix("").as_posix()

    def _text_directory(self) -> Path:
        directory = Config().novelai_text_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"NovelAI text directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"NovelAI text directory is not a directory: {directory}"
            )
        return directory.resolve()

    def _validate_id(self, identifier: str) -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid novelai-text id: {identifier}")

    def _to_text_spec(self, data: dict[str, Any], identifier: str) -> NovelAiTextSpec:
        text = str(data.get("input") or "").strip()
        if not text:
            raise ValueError("input is empty or missing.")
        model = str(data.get("model") or "").strip() or DEFAULT_MODEL
        return NovelAiTextSpec(
            id=identifier,
            input=text,
            model=model,
            max_length=(
                DEFAULT_MAX_LENGTH
                if data.get("max_length") is None
                else int(data["max_length"])
            ),
        )
