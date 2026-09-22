from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.novelai.text.novelai_lorebook import NovelAiLorebook
from ai_media_generation.domain.novelai.text.novelai_text_spec import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    NovelAiTextSpec,
)
from ai_media_generation.repository.json_io import (
    NOVELAI_TEXT_LOREBOOK_SCHEMA,
    NOVELAI_TEXT_SPEC_SCHEMA,
    read_json,
    to_string_tuple,
)
from ai_media_generation.repository.novelai.text_output_repository import (
    NovelAiTextOutputRepository,
)


class NovelAiTextSpecRepository:
    def get(self, ids: tuple[str, ...] = ()) -> tuple[NovelAiTextSpec, ...]:
        config = Config()
        directory = self._text_directory()
        paths = self._paths_for(directory, ids) if ids else self._json_paths(directory)
        lorebooks = self._lorebooks(config.novelai_text_lorebook_directory)
        return tuple(
            self._to_text_spec(
                read_json(path, NOVELAI_TEXT_SPEC_SCHEMA),
                path,
                self._id_for(directory, path),
                lorebooks,
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

    def _validate_id(self, identifier: str, label: str = "novelai-text") -> None:
        path = Path(identifier)
        if (
            not identifier
            or path.is_absolute()
            or not path.parts
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid {label} id: {identifier}")

    def _to_text_spec(
        self,
        data: dict[str, Any],
        path: Path,
        identifier: str,
        lorebooks: tuple[NovelAiLorebook, ...],
    ) -> NovelAiTextSpec:
        text_path = path.with_suffix(".txt")
        if not text_path.is_file():
            raise FileNotFoundError(
                f"NovelAI text opening not found: {identifier}.txt"
            )
        text = text_path.read_text(encoding="utf-8").lstrip()
        if not text.strip():
            raise ValueError(f"NovelAI text {identifier} opening is empty.")
        model = str(data.get("model") or "").strip() or DEFAULT_MODEL
        output = str(data.get("output") or "").strip().replace("\\", "/")
        if output:
            self._validate_id(output, "output")
            output = Path(output).as_posix()
        return NovelAiTextSpec(
            id=identifier,
            input=text,
            model=model,
            max_length=(
                DEFAULT_MAX_LENGTH
                if data.get("max_length") is None
                else int(data["max_length"])
            ),
            stop=to_string_tuple(data.get("stop")),
            system_prompt=self._system_prompt(model),
            memory=self._memory(model),
            lorebooks=lorebooks,
            output=output,
            previous=(
                NovelAiTextOutputRepository().texts(output)
                if output
                else ()
            ),
        )

    def _lorebooks(self, directory: Path) -> tuple[NovelAiLorebook, ...]:
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise NotADirectoryError(
                f"NovelAI lorebook is not a directory: {directory}"
            )
        return tuple(
            self._to_lorebook(directory, path)
            for path in sorted(
                path for path in directory.rglob("*.json") if path.is_file()
            )
        )

    def _to_lorebook(self, directory: Path, path: Path) -> NovelAiLorebook:
        identifier = self._id_for(directory, path)
        self._validate_id(identifier, "lorebook")
        data = read_json(path, NOVELAI_TEXT_LOREBOOK_SCHEMA)
        text_path = path.with_suffix(".txt")
        if not text_path.is_file():
            raise FileNotFoundError(
                f"NovelAI lorebook text not found: {identifier}.txt"
            )
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"NovelAI lorebook {identifier} text is empty.")
        return NovelAiLorebook(
            id=identifier,
            text=text,
            keys=to_string_tuple(data.get("keys")),
        )

    def _memory(self, model: str) -> str:
        return self._context_text(
            Config().novelai_text_memory(model),
            "memory",
        )

    def _system_prompt(self, model: str) -> str:
        return self._context_text(
            Config().novelai_text_system_prompt(model),
            "system_prompt",
        )

    def _context_text(self, path: Path, label: str) -> str:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            return ""
        if not resolved.is_file():
            raise ValueError(f"NovelAI {label} is not a file: {resolved}")
        return resolved.read_text(encoding="utf-8").strip()
