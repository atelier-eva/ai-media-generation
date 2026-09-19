from argparse import ArgumentParser

from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.repository.model_repository import ModelRepository


def add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "Forward localhost:8188 over Direct SSH to the running RunPod pod, "
            "and generate through that tunnel."
        ),
    )


def require_remote_models(url: str, profile: str) -> None:
    repository = ModelRepository()
    ComfyUi.require_filenames(
        url,
        repository.filenames_by_folder(repository.get((profile,))),
        f"Run: ai-media-generation pod-sync-models --profile {profile}",
    )
