from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.subject.feature.subject_feature import (
    SubjectFeature,
    SubjectFeaturePolarity,
)
from ai_media_generation.domain.animagine.lora_dataset.subject.subject import Subject
from ai_media_generation.repository.json_io import (
    ANIMAGINE_LORA_SCHEMAS,
    read_json,
    to_string_tuple,
)


class SubjectRepository:
    def find(self) -> tuple[Subject, ...]:
        directory = Config().animagine_lora_training_characters_directory
        subjects: list[Subject] = []
        names: dict[str, Path] = {}
        for path in self._json_paths(directory):
            subject = self._to_subject(
                read_json(path, ANIMAGINE_LORA_SCHEMAS["characters"])
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
        features = tuple(
            self._to_feature(entry, SubjectFeaturePolarity.POSITIVE)
            for entry in data.get("positive") or []
        ) + tuple(
            SubjectFeature(tag, SubjectFeaturePolarity.NEGATIVE)
            for tag in to_string_tuple(data.get("negative"))
        )
        return Subject(
            name=data["name"].strip(),
            kinds=self._to_kinds(data.get("subject")),
            features=features,
        )

    def _to_feature(
        self, entry: Any, polarity: SubjectFeaturePolarity
    ) -> SubjectFeature:
        if isinstance(entry, str):
            return SubjectFeature(entry.strip(), polarity)
        skip = entry.get("skip_camera") or {}
        return SubjectFeature(
            entry["tag"].strip(),
            polarity,
            weight=self._to_weight(entry.get("weight")),
            skip_angles=to_string_tuple(skip.get("angle")),
            skip_distances=to_string_tuple(skip.get("distance")),
        )

    def _to_weight(self, value: Any) -> float | None:
        if value is None:
            return None
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"Feature weight must be greater than 0: {weight}")
        return weight

    def _to_kinds(self, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            kind = value.strip()
            return (kind,) if kind else ()
        return to_string_tuple(value)
