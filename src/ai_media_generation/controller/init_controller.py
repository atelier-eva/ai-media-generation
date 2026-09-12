from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
from sys import argv

from ai_media_generation.config import Config

_JSON_FILES = (
    "art-style.json",
    "camera.json",
    "expression.json",
    "generation.json",
    "pose.json",
    "scene.json",
)
_CHARACTERS_DIRECTORY = "characters"


class InitController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--directory",
            default=".",
            help=(
                "Parent directory for feature input folders. "
                f"LoRA training spec templates go in {Config.LORA_TRAINING_SPEC_DIRECTORY}/. "
                f"Animagine spec templates go in {Config.ANIMAGINE_SPEC_DIRECTORY}/. "
                f"Qwen spec templates go in {Config.QWEN_SPEC_DIRECTORY}/. "
                f"Qwen LoRA training spec templates go in {Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY}/. "
                f"Kagee spec templates go in {Config.KAGEE_SPEC_DIRECTORY}/. "
                f"Music spec templates go in {Config.MUSIC_SPEC_DIRECTORY}/. "
                "Defaults to the current directory."
            ),
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing JSON files.",
        )
        args = parser.parse_args(argv[2:])
        text = args.directory.strip()
        if not text:
            raise ValueError("--directory is empty.")
        path = Path(text).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        lora_training = path / Config.LORA_TRAINING_SPEC_DIRECTORY
        animagine = path / Config.ANIMAGINE_SPEC_DIRECTORY
        qwen = path / Config.QWEN_SPEC_DIRECTORY
        qwen_lora_training = path / Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY
        kagee = path / Config.KAGEE_SPEC_DIRECTORY
        music = path / Config.MUSIC_SPEC_DIRECTORY
        lora_training.mkdir(parents=True, exist_ok=True)
        for name in _JSON_FILES:
            self._write_json(lora_training / name, args.force)
        self._write_characters(lora_training / _CHARACTERS_DIRECTORY, args.force)
        self._write_directory(
            (Config.ANIMAGINE_SPEC_DIRECTORY,),
            animagine,
            args.force,
        )
        self._write_directory(
            (Config.QWEN_SPEC_DIRECTORY,),
            qwen,
            args.force,
        )
        qwen_lora_training.mkdir(parents=True, exist_ok=True)
        for name in _JSON_FILES:
            self._write_resource(
                (Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY, name),
                qwen_lora_training / name,
                args.force,
            )
        self._write_directory(
            (Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY, _CHARACTERS_DIRECTORY),
            qwen_lora_training / _CHARACTERS_DIRECTORY,
            args.force,
        )
        self._write_directory(
            (Config.KAGEE_SPEC_DIRECTORY,),
            kagee,
            args.force,
        )
        self._write_directory(
            (Config.MUSIC_SPEC_DIRECTORY,),
            music,
            args.force,
        )
        self._write_env()
        print(f"LoRA training spec directory: {lora_training}")
        print(f"Animagine spec directory: {animagine}")
        print(f"Qwen spec directory: {qwen}")
        print(f"Qwen LoRA training spec directory: {qwen_lora_training}")
        print(f"Kagee spec directory: {kagee}")
        print(f"Music spec directory: {music}")
        print(
            "Fill in the JSON, then run: "
            "ai-media-generation lora-training, animagine, qwen, "
            "qwen-lora-training, kagee, music, or report"
        )

    def _write_env(self) -> None:
        env_path = Path(".env")
        if env_path.exists():
            print(f"Skipped existing: {env_path.resolve()}")
            return
        example = files("ai_media_generation.resources").joinpath("env.example")
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created: {env_path.resolve()}")

    def _write_characters(self, destination: Path, force: bool) -> None:
        self._write_directory(
            (Config.LORA_TRAINING_SPEC_DIRECTORY, destination.name),
            destination,
            force,
        )

    def _write_directory(
        self, relative: tuple[str, ...], destination: Path, force: bool
    ) -> None:
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(f"Not a directory: {destination}")
        existed = destination.exists()
        destination.mkdir(parents=True, exist_ok=True)
        source_dir = files("ai_media_generation.resources").joinpath(*relative)
        names = tuple(
            sorted(
                item.name
                for item in source_dir.iterdir()
                if item.name.endswith(".json") and item.is_file()
            )
        )
        if not names:
            action = "Skipped existing" if existed else "Created"
            print(f"{action}: {destination}")
            return
        for name in names:
            self._write_resource((*relative, name), destination / name, force)

    def _write_json(self, destination: Path, force: bool) -> None:
        self._write_resource(
            (Config.LORA_TRAINING_SPEC_DIRECTORY, destination.name),
            destination,
            force,
        )

    def _write_resource(
        self, relative: tuple[str, ...], destination: Path, force: bool
    ) -> None:
        existed = destination.exists()
        if existed and not force:
            print(f"Skipped existing: {destination}")
            return
        source = files("ai_media_generation.resources").joinpath(*relative)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        action = "Overwrote" if existed else "Created"
        print(f"{action}: {destination}")
