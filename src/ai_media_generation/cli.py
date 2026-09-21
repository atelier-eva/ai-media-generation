"""Command-line entry point for the AI media generation assistant."""

from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from sys import argv, stderr

from dotenv import load_dotenv

from ai_media_generation.controller.generate_anima_controller import (
    GenerateAnimaController,
)
from ai_media_generation.controller.generate_anima_lora_training_images_controller import (
    GenerateAnimaLoraTrainingImagesController,
)
from ai_media_generation.controller.generate_animagine_controller import (
    GenerateAnimagineController,
)
from ai_media_generation.controller.generate_animagine_lora_training_images_controller import (
    GenerateAnimagineLoraTrainingImagesController,
)
from ai_media_generation.controller.generate_music_controller import (
    GenerateMusicController,
)
from ai_media_generation.controller.generate_qwen_controller import (
    GenerateQwenController,
)
from ai_media_generation.controller.generate_qwen_edit_controller import (
    GenerateQwenEditController,
)
from ai_media_generation.controller.generate_qwen_lora_training_images_controller import (
    GenerateQwenLoraTrainingImagesController,
)
from ai_media_generation.controller.init_controller import InitController
from ai_media_generation.controller.pod_connect_controller import PodConnectController
from ai_media_generation.controller.pod_start_controller import PodStartController
from ai_media_generation.controller.pod_status_controller import PodStatusController
from ai_media_generation.controller.pod_stop_controller import PodStopController
from ai_media_generation.controller.pod_sync_models_controller import (
    PodSyncModelsController,
)
from ai_media_generation.controller.report_anima_lora_training_patterns_controller import (
    ReportAnimaLoraTrainingPatternsController,
)
from ai_media_generation.controller.report_animagine_lora_training_patterns_controller import (
    ReportAnimagineLoraTrainingPatternsController,
)
from ai_media_generation.infrastructure.error import InfrastructureError

_USER_ERRORS = (
    FileNotFoundError,
    InfrastructureError,
    NotADirectoryError,
    ValueError,
)


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    try:
        _run()
    except _USER_ERRORS as error:
        print(f"ai-media-generation: error: {error}", file=stderr)
        raise SystemExit(1) from error


def _run() -> None:
    arguments = argv[1:]
    command = arguments[0] if arguments else ""
    if command == "init":
        InitController().execute(_command_parser("init"))
        return
    if command == "animagine-lora-training":
        GenerateAnimagineLoraTrainingImagesController().execute(
            _command_parser("animagine-lora-training")
        )
        return
    if command == "animagine":
        GenerateAnimagineController().execute(_command_parser("animagine"))
        return
    if command == "qwen":
        GenerateQwenController().execute(_command_parser("qwen"))
        return
    if command == "anima":
        GenerateAnimaController().execute(_command_parser("anima"))
        return
    if command == "anima-lora-report":
        ReportAnimaLoraTrainingPatternsController().execute(
            _command_parser("anima-lora-report")
        )
        return
    if command == "anima-lora-training":
        GenerateAnimaLoraTrainingImagesController().execute(
            _command_parser("anima-lora-training")
        )
        return
    if command == "qwen-edit":
        GenerateQwenEditController().execute(_command_parser("qwen-edit"))
        return
    if command == "qwen-lora-training":
        GenerateQwenLoraTrainingImagesController().execute(
            _command_parser("qwen-lora-training")
        )
        return
    if command == "music":
        GenerateMusicController().execute(_command_parser("music"))
        return
    if command == "pod-connect":
        PodConnectController().execute(_command_parser("pod-connect"))
        return
    if command == "pod-start":
        PodStartController().execute(_command_parser("pod-start"))
        return
    if command == "pod-status":
        PodStatusController().execute(_command_parser("pod-status"))
        return
    if command == "pod-stop":
        PodStopController().execute(_command_parser("pod-stop"))
        return
    if command == "pod-sync-models":
        PodSyncModelsController().execute(_command_parser("pod-sync-models"))
        return
    if command == "report":
        ReportAnimagineLoraTrainingPatternsController().execute(
            _command_parser("report")
        )
        return
    if command == "see-through":
        print(
            "ai-media-generation: see-through was removed.\n"
            "Use the official See-through CLI instead:\n"
            "  python inference/scripts/inference_psd.py --srcp IMAGE --save_to_psd",
            file=stderr,
        )
        raise SystemExit(1)
    _parser().parse_args(arguments)


def _command_parser(command: str) -> ArgumentParser:
    return ArgumentParser(prog=f"ai-media-generation {command}")


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="ai-media-generation")
    parser.add_argument(
        "--version",
        action="version",
        version=f"ai-media-generation {version('ai-media-generation')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "animagine-lora-training",
        help="Generate LoRA training images with Animagine XL 4.0.",
    )
    subparsers.add_parser(
        "animagine",
        help="Generate images from prompt JSON specs with Animagine XL 4.0 (nested folders allowed).",
    )
    subparsers.add_parser(
        "qwen",
        help="Generate images from qwen JSON specs with Qwen-Image-2512 (nested folders allowed).",
    )
    subparsers.add_parser(
        "anima",
        help="Generate images from anima JSON specs with Anima Aesthetic v1.1 (nested folders allowed).",
    )
    subparsers.add_parser(
        "anima-lora-report",
        help="Write Anima LoRA training pattern rows to CSV.",
    )
    subparsers.add_parser(
        "anima-lora-training",
        help="Generate LoRA training images with Anima Aesthetic v1.1.",
    )
    subparsers.add_parser(
        "qwen-edit",
        help="Generate images from qwen-edit JSON specs with Qwen-Image-Edit-2511 (nested folders allowed).",
    )
    subparsers.add_parser(
        "qwen-lora-training",
        help="Generate LoRA training images with Qwen-Image-Edit-2511.",
    )
    subparsers.add_parser(
        "music",
        help="Generate music from music/*.json specs.",
    )
    subparsers.add_parser(
        "init",
        help="Create input JSON templates.",
    )
    subparsers.add_parser(
        "pod-connect",
        help="Forward localhost:8188 to the running pod over Direct SSH.",
    )
    subparsers.add_parser(
        "pod-start",
        help="Start the RunPod pod and wait until RUNNING.",
    )
    subparsers.add_parser(
        "pod-status",
        help="Show RunPod pod status and Direct SSH from RUNPOD_POD_ID.",
    )
    subparsers.add_parser(
        "pod-stop",
        help="Stop the RunPod pod and wait until EXITED.",
    )
    subparsers.add_parser(
        "pod-sync-models",
        help="Download required models onto the running RunPod pod.",
    )
    subparsers.add_parser(
        "report",
        help="Write Animagine LoRA training pattern rows to CSV.",
    )
    return parser
