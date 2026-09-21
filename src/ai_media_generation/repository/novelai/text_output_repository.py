import re
from pathlib import Path

from ai_media_generation.config import Config

_INDEXED_TEXT = re.compile(r"^(\d+)\.txt$")
_WIDTH = 3


def _indexed_number(name: str) -> int | None:
    match = _INDEXED_TEXT.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def _next_indexed_stem(names: tuple[str, ...]) -> str:
    numbers = tuple(
        number
        for name in names
        if (number := _indexed_number(name)) is not None
    )
    return f"{max(numbers, default=0) + 1:0{_WIDTH}d}"


class NovelAiTextOutputRepository:
    def next_filename_prefix(self, output: str) -> str:
        files = self._indexed_files(self._directory(output))
        return f"{output}/{_next_indexed_stem(tuple(path.name for path in files))}"

    def texts(self, output: str) -> tuple[str, ...]:
        return tuple(
            path.read_text(encoding="utf-8")
            for path in self._indexed_files(self._directory(output))
        )

    def _root(self) -> Path:
        return Config().novelai_text_output_directory

    def _directory(self, output: str) -> Path:
        root = self._root().expanduser().resolve()
        path = Path(output)
        if (
            not output
            or path.is_absolute()
            or not path.parts
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"Invalid output id: {output}")
        directory = (root / output).expanduser().resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"Invalid output id: {output}")
        return directory

    def _indexed_files(self, directory: Path) -> tuple[Path, ...]:
        if not directory.exists():
            return ()
        if not directory.is_dir():
            raise NotADirectoryError(
                f"NovelAI text output is not a directory: {directory}"
            )
        files: list[tuple[int, Path]] = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            number = _indexed_number(path.name)
            if number is None:
                continue
            files.append((number, path))
        files.sort(key=lambda item: item[0])
        return tuple(path for _, path in files)
