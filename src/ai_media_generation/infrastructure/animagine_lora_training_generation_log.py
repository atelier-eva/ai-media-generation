import json
from dataclasses import dataclass
from typing import Any

from ai_media_generation.config import Config

_OPTIONAL_NAMES = (
    "expression",
    "pose",
    "art_style",
    "background",
    "lighting",
)
_REQUIRED_NAMES = ("subject", "angle", "distance")
_FIELDS = (
    *_REQUIRED_NAMES,
    *_OPTIONAL_NAMES,
    "row",
    "seed",
    "batch_size",
    "files",
)


class AnimagineLoraTrainingGenerationLog:
    @dataclass(frozen=True)
    class _Record:
        subject: str
        angle: str
        distance: str
        expression: str | None
        pose: str | None
        art_style: str | None
        background: str | None
        lighting: str | None
        row: int
        seed: int
        batch_size: int
        files: tuple[str, ...]

        @property
        def identity(
            self,
        ) -> tuple[
            str, str, str, str | None, str | None, str | None, str | None, str | None
        ]:
            return (
                self.subject,
                self.angle,
                self.distance,
                self.expression,
                self.pose,
                self.art_style,
                self.background,
                self.lighting,
            )

    def __init__(self) -> None:
        self._path = Config().animagine_lora_training_generations_jsonl

    def append(
        self,
        *,
        subject: str,
        angle: str,
        distance: str,
        expression: str | None,
        pose: str | None,
        art_style: str | None,
        background: str | None,
        lighting: str | None,
        row: int,
        seed: int,
        batch_size: int,
        files: tuple[str, ...],
    ) -> None:
        record = self._record(
            {
                "subject": subject,
                "angle": angle,
                "distance": distance,
                "expression": expression,
                "pose": pose,
                "art_style": art_style,
                "background": background,
                "lighting": lighting,
                "row": row,
                "seed": seed,
                "batch_size": batch_size,
                "files": list(files),
            }
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(self._to_line(record))
            file.write("\n")

    def contains(
        self,
        *,
        subject: str,
        angle: str,
        distance: str,
        expression: str | None,
        pose: str | None,
        art_style: str | None,
        background: str | None,
        lighting: str | None,
        seed: int,
    ) -> bool:
        identity = (
            subject,
            angle,
            distance,
            expression,
            pose,
            art_style,
            background,
            lighting,
        )
        return any(
            record.identity == identity and record.seed == seed
            for record in self._records()
        )

    def _to_line(self, record: "_Record") -> str:
        return json.dumps(
            {
                "subject": record.subject,
                "angle": record.angle,
                "distance": record.distance,
                "expression": record.expression,
                "pose": record.pose,
                "art_style": record.art_style,
                "background": record.background,
                "lighting": record.lighting,
                "row": record.row,
                "seed": record.seed,
                "batch_size": record.batch_size,
                "files": list(record.files),
            },
            ensure_ascii=False,
        )

    def _error(self, message: str, line_number: int | None = None) -> ValueError:
        location = f": line {line_number}" if line_number is not None else ""
        return ValueError(f"Invalid JSONL: {self._path}{location}: {message}")

    def _int_field(
        self,
        data: dict[str, Any],
        name: str,
        *,
        minimum: int,
        line_number: int | None,
    ) -> int:
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._error(f"{name} must be an integer.", line_number)
        if value < minimum:
            raise self._error(f"{name} must be >= {minimum}.", line_number)
        return value

    def _str_field(self, value: Any, name: str, line_number: int | None) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self._error(f"{name} must be a non-empty string.", line_number)
        return value.strip()

    def _optional_str_field(
        self, value: Any, name: str, line_number: int | None
    ) -> str | None:
        if value is None:
            return None
        return self._str_field(value, name, line_number)

    def _record(
        self, data: dict[str, Any], line_number: int | None = None
    ) -> _Record:
        extra = sorted(set(data) - set(_FIELDS))
        if extra:
            raise self._error(f"Unknown fields: {', '.join(extra)}.", line_number)
        missing = [name for name in _FIELDS if name not in data]
        if missing:
            raise self._error(f"Missing fields: {', '.join(missing)}.", line_number)
        files = data["files"]
        if not isinstance(files, list) or any(
            not isinstance(item, str) or not item.strip() for item in files
        ):
            raise self._error(
                "files must be an array of non-empty strings.", line_number
            )
        return self._Record(
            subject=self._str_field(data["subject"], "subject", line_number),
            angle=self._str_field(data["angle"], "angle", line_number),
            distance=self._str_field(data["distance"], "distance", line_number),
            expression=self._optional_str_field(
                data["expression"], "expression", line_number
            ),
            pose=self._optional_str_field(data["pose"], "pose", line_number),
            art_style=self._optional_str_field(data["art_style"], "art_style", line_number),
            background=self._optional_str_field(
                data["background"], "background", line_number
            ),
            lighting=self._optional_str_field(data["lighting"], "lighting", line_number),
            row=self._int_field(data, "row", minimum=1, line_number=line_number),
            seed=self._int_field(data, "seed", minimum=0, line_number=line_number),
            batch_size=self._int_field(
                data, "batch_size", minimum=1, line_number=line_number
            ),
            files=tuple(files),
        )

    def _records(self) -> tuple[_Record, ...]:
        path = self._path
        if not path.exists():
            return ()
        if not path.is_file():
            raise self._error("path is not a file.")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise self._error(str(error)) from error
        records: list[AnimagineLoraTrainingGenerationLog._Record] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as error:
                raise self._error(error.msg, line_number) from error
            if not isinstance(loaded, dict):
                raise self._error("JSON must be an object.", line_number)
            records.append(self._record(loaded, line_number))
        return tuple(records)
