from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.music_spec.music_spec import MusicSpec
from ai_media_generation.repository.json_io import MUSIC_SCHEMA, read_json


class MusicSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[MusicSpec, ...]:
        paths = self._paths_for(ids) if ids else self._json_paths()
        return tuple(
            self._to_music_spec(read_json(path, MUSIC_SCHEMA), path.stem)
            for path in paths
        )

    def _json_paths(self) -> tuple[Path, ...]:
        directory = self._music_directory()
        paths = tuple(
            sorted(path for path in directory.glob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No music JSON in {directory}")
        return paths

    def _music_directory(self) -> Path:
        directory = Config().music_spec_directory
        if not directory.exists():
            raise FileNotFoundError(f"Music directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Music directory is not a directory: {directory}"
            )
        return directory

    def _paths_for(self, ids: tuple[str, ...]) -> tuple[Path, ...]:
        directory = self._music_directory()
        paths: list[Path] = []
        for identifier in ids:
            self._validate_id(identifier)
            path = directory / f"{identifier}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Music JSON not found: {path.name}")
            paths.append(path)
        return tuple(paths)

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _to_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _to_music_spec(self, data: dict[str, Any], identifier: str) -> MusicSpec:
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is empty or missing.")
        language = str(data.get("vocal_language") or "").strip()
        return MusicSpec(
            id=identifier,
            prompt=prompt,
            lyrics=str(data.get("lyrics") or "").strip(),
            vocal_language=language or None,
            duration=self._to_float(data.get("duration")),
            bpm=self._to_int(data.get("bpm")),
        )

    def _validate_id(self, identifier: str) -> None:
        if not identifier or Path(identifier).name != identifier:
            raise ValueError(f"Invalid music id: {identifier}")
