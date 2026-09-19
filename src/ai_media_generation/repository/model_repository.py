from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.repository.json_io import read_resource_json

_PRECISIONS = frozenset({"fp8", "bf16"})


@dataclass(frozen=True)
class SyncModel:
    repo: str
    hub_path: str
    path: str
    profiles: frozenset[str]
    precision: str | None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def folder(self) -> str:
        return Path(self.path).parts[1]


class ModelRepository:
    def get(self, profiles: tuple[str, ...] = ()) -> tuple[SyncModel, ...]:
        loaded = read_resource_json("models.json")
        raw = loaded.get("models")
        if not isinstance(raw, list) or not raw:
            raise ValueError("models.json did not list models.")
        models = tuple(self._model(item) for item in raw)
        known = frozenset(profile for model in models for profile in model.profiles)
        extra = sorted(set(profiles) - known)
        if extra:
            raise ValueError(f"Unknown profile: {extra[0]}.")
        if profiles:
            models = tuple(
                model for model in models if model.profiles.intersection(profiles)
            )
            if not models:
                raise ValueError("No models match the given profile(s).")
        precision = Config().qwen_precision
        models = tuple(
            model
            for model in models
            if model.precision is None or model.precision == precision
        )
        if not models:
            raise ValueError("No models match the given profile(s).")
        return models

    def filenames_by_folder(
        self, models: tuple[SyncModel, ...]
    ) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for model in models:
            grouped.setdefault(model.folder, []).append(model.name)
        return {folder: tuple(names) for folder, names in grouped.items()}

    def diffusion_model_name(self, profile: str) -> str:
        models = tuple(
            model
            for model in self.get((profile,))
            if model.folder == "diffusion_models"
        )
        if len(models) != 1:
            raise ValueError(
                f"models.json must list exactly one diffusion model for {profile}."
            )
        return models[0].name

    def _model(self, item: Any) -> SyncModel:
        if not isinstance(item, dict):
            raise ValueError("models.json entry must be an object.")
        repo = self._text(item, "repo")
        hub_path = self._text(item, "hub_path")
        path = self._text(item, "path")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid model path: {path}")
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "models" or not parts[1] or not parts[2]:
            raise ValueError(f"Invalid model path: {path}")
        profiles = item.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError(f"models.json profiles missing for {path}.")
        names = frozenset(self._profile(value) for value in profiles)
        return SyncModel(
            repo=repo,
            hub_path=hub_path,
            path=path,
            profiles=names,
            precision=self._precision(item, path),
        )

    def _precision(self, item: dict[str, Any], path: str) -> str | None:
        value = item.get("precision")
        if value is None:
            return None
        if not isinstance(value, str) or value.strip() not in _PRECISIONS:
            raise ValueError(f"models.json precision must be fp8 or bf16 for {path}.")
        return value.strip()

    def _profile(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("models.json profile must be a non-empty string.")
        return value.strip()

    def _text(self, item: dict[str, Any], name: str) -> str:
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"models.json {name} must be a non-empty string.")
        return value.strip()
