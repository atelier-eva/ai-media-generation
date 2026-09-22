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


def _resource_segments(*parts: str) -> tuple[str, ...]:
    segments: list[str] = []
    for part in parts:
        segments.extend(segment for segment in part.split("/") if segment)
    return tuple(segments)


class InitController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--directory",
            default=".",
            help=(
                "Parent directory for feature input folders. "
                f"Animagine LoRA training spec templates go in {Config.ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY}/. "
                f"Animagine spec templates go in {Config.ANIMAGINE_SPEC_DIRECTORY}/. "
                f"Qwen spec templates go in {Config.QWEN_SPEC_DIRECTORY}/. "
                f"Anima spec templates go in {Config.ANIMA_SPEC_DIRECTORY}/. "
                f"NovelAI spec templates go in {Config.NOVELAI_SPEC_DIRECTORY}/. "
                f"NovelAI text spec templates go in {Config.NOVELAI_TEXT_SPEC_DIRECTORY}/. "
                f"NovelAI text system prompt goes in "
                f"{(Path(Config.NOVELAI_TEXT_SPEC_DIRECTORY).parent / Config.NOVELAI_TEXT_SYSTEM_PROMPT).as_posix()}. "
                f"NovelAI text memory goes in "
                f"{(Path(Config.NOVELAI_TEXT_SPEC_DIRECTORY).parent / Config.NOVELAI_TEXT_MEMORY).as_posix()}. "
                f"NovelAI text lorebook templates go in "
                f"{(Path(Config.NOVELAI_TEXT_SPEC_DIRECTORY).parent / Config.NOVELAI_TEXT_LOREBOOK_DIRECTORY).as_posix()}/. "
                f"Anima LoRA training spec templates go in {Config.ANIMA_LORA_TRAINING_SPEC_DIRECTORY}/. "
                f"Qwen edit spec templates go in {Config.QWEN_EDIT_SPEC_DIRECTORY}/. "
                f"Qwen LoRA training spec templates go in {Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY}/. "
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
        animagine_lora_training = path / Config.ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY
        animagine = path / Config.ANIMAGINE_SPEC_DIRECTORY
        qwen = path / Config.QWEN_SPEC_DIRECTORY
        anima = path / Config.ANIMA_SPEC_DIRECTORY
        novelai = path / Config.NOVELAI_SPEC_DIRECTORY
        novelai_text = path / Config.NOVELAI_TEXT_SPEC_DIRECTORY
        anima_lora_training = path / Config.ANIMA_LORA_TRAINING_SPEC_DIRECTORY
        qwen_edit = path / Config.QWEN_EDIT_SPEC_DIRECTORY
        qwen_lora_training = path / Config.QWEN_LORA_TRAINING_SPEC_DIRECTORY
        music = path / Config.MUSIC_SPEC_DIRECTORY
        animagine_lora_training.mkdir(parents=True, exist_ok=True)
        for name in _JSON_FILES:
            self._write_resource(
                (Config.ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY, name),
                animagine_lora_training / name,
                args.force,
            )
        self._write_directory(
            (Config.ANIMAGINE_LORA_TRAINING_SPEC_DIRECTORY, _CHARACTERS_DIRECTORY),
            animagine_lora_training / _CHARACTERS_DIRECTORY,
            args.force,
        )
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
        self._write_directory(
            (Config.ANIMA_SPEC_DIRECTORY,),
            anima,
            args.force,
        )
        self._write_directory(
            (Config.NOVELAI_SPEC_DIRECTORY,),
            novelai,
            args.force,
        )
        self._write_directory(
            (Config.NOVELAI_TEXT_SPEC_DIRECTORY,),
            novelai_text,
            args.force,
        )
        text_root = novelai_text.parent
        self._write_resource(
            ("novelai", "text", Config.NOVELAI_TEXT_SYSTEM_PROMPT),
            text_root / Config.NOVELAI_TEXT_SYSTEM_PROMPT,
            args.force,
        )
        self._write_resource(
            ("novelai", "text", Config.NOVELAI_TEXT_MEMORY),
            text_root / Config.NOVELAI_TEXT_MEMORY,
            args.force,
        )
        novelai_text_lorebook = text_root / Config.NOVELAI_TEXT_LOREBOOK_DIRECTORY
        self._write_directory(
            ("novelai", "text", Config.NOVELAI_TEXT_LOREBOOK_DIRECTORY),
            novelai_text_lorebook,
            args.force,
        )
        anima_lora_training.mkdir(parents=True, exist_ok=True)
        for name in _JSON_FILES:
            self._write_resource(
                (Config.ANIMA_LORA_TRAINING_SPEC_DIRECTORY, name),
                anima_lora_training / name,
                args.force,
            )
        self._write_directory(
            (Config.ANIMA_LORA_TRAINING_SPEC_DIRECTORY, _CHARACTERS_DIRECTORY),
            anima_lora_training / _CHARACTERS_DIRECTORY,
            args.force,
        )
        self._write_directory(
            (Config.QWEN_EDIT_SPEC_DIRECTORY,),
            qwen_edit,
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
            (Config.MUSIC_SPEC_DIRECTORY,),
            music,
            args.force,
        )
        self._write_env()
        print(f"Animagine LoRA training spec directory: {animagine_lora_training}")
        print(f"Animagine spec directory: {animagine}")
        print(f"Qwen spec directory: {qwen}")
        print(f"Anima spec directory: {anima}")
        print(f"NovelAI spec directory: {novelai}")
        print(f"NovelAI text spec directory: {novelai_text}")
        print(
            "NovelAI text system prompt: "
            f"{text_root / Config.NOVELAI_TEXT_SYSTEM_PROMPT}"
        )
        print(
            f"NovelAI text memory: {text_root / Config.NOVELAI_TEXT_MEMORY}"
        )
        print(f"NovelAI text lorebook directory: {novelai_text_lorebook}")
        print(f"Anima LoRA training spec directory: {anima_lora_training}")
        print(f"Qwen edit spec directory: {qwen_edit}")
        print(f"Qwen LoRA training spec directory: {qwen_lora_training}")
        print(f"Music spec directory: {music}")
        print(
            "Fill in the JSON, then run: "
            "ai-media-generation animagine-lora-training, animagine, qwen, anima, "
            "novelai, novelai-text, anima-lora-report, anima-lora-training, "
            "qwen-edit, qwen-lora-training, music, or report"
        )

    def _write_env(self) -> None:
        env_path = Path(".env")
        if env_path.exists():
            print(f"Skipped existing: {env_path.resolve()}")
            return
        example = files("ai_media_generation.resources").joinpath("env.example")
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created: {env_path.resolve()}")

    def _write_directory(
        self, relative: tuple[str, ...], destination: Path, force: bool
    ) -> None:
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(f"Not a directory: {destination}")
        existed = destination.exists()
        destination.mkdir(parents=True, exist_ok=True)
        source_dir = files("ai_media_generation.resources").joinpath(
            *_resource_segments(*relative)
        )
        names = tuple(
            sorted(
                item.name
                for item in source_dir.iterdir()
                if item.is_file()
                and (item.name.endswith(".json") or item.name.endswith(".txt"))
            )
        )
        if not names:
            action = "Skipped existing" if existed else "Created"
            print(f"{action}: {destination}")
            return
        for name in names:
            self._write_resource((*relative, name), destination / name, force)

    def _write_resource(
        self, relative: tuple[str, ...], destination: Path, force: bool
    ) -> None:
        existed = destination.exists()
        if existed and not force:
            print(f"Skipped existing: {destination}")
            return
        source = files("ai_media_generation.resources").joinpath(
            *_resource_segments(*relative)
        )
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        action = "Overwrote" if existed else "Created"
        print(f"{action}: {destination}")
