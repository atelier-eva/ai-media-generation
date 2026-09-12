"""Command-line entry point for the AI media generation assistant."""

from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from sys import argv, stderr

from dotenv import load_dotenv

from ai_media_generation.controller.generate_animagine_controller import (
    GenerateAnimagineController,
)
from ai_media_generation.controller.generate_kagee_controller import (
    GenerateKageeController,
)
from ai_media_generation.controller.generate_lora_training_images_controller import (
    GenerateLoraTrainingImagesController,
)
from ai_media_generation.controller.generate_music_controller import (
    GenerateMusicController,
)
from ai_media_generation.controller.generate_qwen_controller import (
    GenerateQwenController,
)
from ai_media_generation.controller.generate_qwen_lora_training_images_controller import (
    GenerateQwenLoraTrainingImagesController,
)
from ai_media_generation.controller.init_controller import InitController
from ai_media_generation.controller.report_lora_training_patterns_controller import (
    ReportLoraTrainingPatternsController,
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
    if command == "lora-training":
        GenerateLoraTrainingImagesController().execute(_command_parser("lora-training"))
        return
    if command == "animagine":
        GenerateAnimagineController().execute(_command_parser("animagine"))
        return
    if command == "qwen":
        GenerateQwenController().execute(_command_parser("qwen"))
        return
    if command == "qwen-lora-training":
        GenerateQwenLoraTrainingImagesController().execute(
            _command_parser("qwen-lora-training")
        )
        return
    if command == "kagee":
        GenerateKageeController().execute(_command_parser("kagee"))
        return
    if command == "music":
        GenerateMusicController().execute(_command_parser("music"))
        return
    if command == "report":
        ReportLoraTrainingPatternsController().execute(_command_parser("report"))
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
        "lora-training",
        help="Generate LoRA training images.",
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
        "qwen-lora-training",
        help="Generate LoRA training images with Qwen-Image-Edit-2511.",
    )
    subparsers.add_parser(
        "kagee",
        help="Convert images to kagee from kagee JSON specs (nested folders allowed).",
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
        "report",
        help="Write LoRA training pattern rows to CSV.",
    )
    return parser
