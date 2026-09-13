from argparse import ArgumentParser

from ai_media_generation.controller.pod_sync_models_controller import (
    filenames_by_folder,
    sync_models,
)
from ai_media_generation.infrastructure.comfy_ui import ComfyUi


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
    ComfyUi.require_filenames(
        url,
        filenames_by_folder(sync_models((profile,))),
        f"Run: ai-media-generation pod-sync-models --profile {profile}",
    )
