from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen.lora_dataset.subject import Subject
from ai_media_generation.repository.json_io import QWEN_LORA_SCHEMAS, read_json


class SubjectRepository:
    def find(self) -> tuple[Subject, ...]:
        directory = Config().qwen_lora_training_characters_directory
        subjects: list[Subject] = []
        names: dict[str, Path] = {}
        for path in self._json_paths(directory):
            subject = self._to_subject(
                read_json(path, QWEN_LORA_SCHEMAS["characters"])
            )
            previous = names.get(subject.name)
            if previous is not None:
                raise ValueError(
                    f"Duplicate character name '{subject.name}': "
                    f"{previous} and {path}"
                )
            names[subject.name] = path
            subjects.append(subject)
        return tuple(subjects)

    def _json_paths(self, directory: Path) -> tuple[Path, ...]:
        if not directory.exists():
            raise FileNotFoundError(f"Characters directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Characters directory is not a directory: {directory}"
            )
        paths = tuple(
            sorted(path for path in directory.glob("*.json") if path.is_file())
        )
        if not paths:
            raise FileNotFoundError(f"No character JSON in {directory}")
        return paths

    def _to_subject(self, data: dict[str, Any]) -> Subject:
        return Subject(
            name=data["name"].strip(),
            images=self._to_images(data.get("images")),
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
                raise FileNotFoundError(
                    f"Qwen LoRA character image not found: {path}"
                )
            paths.append(path)
        if not paths:
            raise ValueError("images is empty or missing.")
        if len(paths) > 3:
            raise ValueError("at most 3 images are supported.")
        return tuple(paths)
