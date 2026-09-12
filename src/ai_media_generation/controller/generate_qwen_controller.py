from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.qwen_spec.get_qwen_specs import GetQwenSpecs
from ai_media_generation.infrastructure.comfy_ui import ComfyUi


class GenerateQwenController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "Qwen JSON paths under qwen/, relative, nested allowed "
                "(e.g. hero/smile.json). Omit to generate every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetQwenSpecs().execute(self._qwen_ids(args.files)).dtos
        if not specs:
            raise ValueError("No qwen JSON to generate.")
        print(f"Processing {len(specs)} qwen JSON file(s).")
        config = Config()
        directory = config.qwen_output_directory
        comfy_ui = ComfyUi()
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = args.base_seed + index
            print(f"[{index + 1}/{len(specs)}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_qwen(
                filename_prefix,
                spec.width,
                spec.height,
                spec.prompt,
                spec.negative,
                seed,
                args.batch_size,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _qwen_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._qwen_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate qwen id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _qwen_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("Qwen id is empty.")
        return text
