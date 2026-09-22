from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.novelai.text.get_novelai_text_specs import (
    GetNovelAiTextSpecs,
)
from ai_media_generation.infrastructure.novelai import NovelAI
from ai_media_generation.repository.novelai.text_output_repository import (
    NovelAiTextOutputRepository,
)


class GenerateNovelAiTextController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "NovelAI text JSON paths under novelai/text/spec/, relative, "
                "nested allowed (e.g. chapter/opening.json). "
                "Each JSON requires a sibling .txt opening. "
                "Omit to generate every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetNovelAiTextSpecs().execute(self._text_ids(args.files)).dtos
        if not specs:
            raise ValueError("No novelai-text JSON to generate.")
        print(f"Processing {len(specs)} novelai-text JSON file(s).")
        config = Config()
        directory = config.novelai_text_output_directory
        novelai = NovelAI()
        outputs = NovelAiTextOutputRepository()
        for index, spec in enumerate(specs):
            filename_prefix = (
                outputs.next_filename_prefix(spec.output)
                if spec.output
                else spec.id
            )
            print(f"[{index + 1}/{len(specs)}] {filename_prefix}")
            texts = novelai.generate_text(
                filename_prefix,
                spec.input,
                spec.model,
                spec.max_length,
            )
            written = novelai.write_text(texts, directory)
            if written:
                print(f"  text: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _text_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._text_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate novelai-text id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _text_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("NovelAI text id is empty.")
        return text
