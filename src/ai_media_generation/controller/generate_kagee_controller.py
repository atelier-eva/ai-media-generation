from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.kagee_spec.get_kagee_specs import GetKageeSpecs
from ai_media_generation.infrastructure.comfy_ui import ComfyUi


class GenerateKageeController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "Kagee JSON paths under kagee/, relative, nested allowed "
                "(e.g. hero/smile.json). Omit to convert every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetKageeSpecs().execute(self._kagee_ids(args.files)).dtos
        if not specs:
            raise ValueError("No kagee JSON to convert.")
        print(f"Processing {len(specs)} kagee JSON file(s).")
        config = Config()
        directory = config.kagee_output_directory
        comfy_ui = ComfyUi()
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = spec.seed if spec.seed is not None else args.base_seed + index
            print(f"[{index + 1}/{len(specs)}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_kagee(
                filename_prefix,
                spec.images,
                spec.prompt,
                seed,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _kagee_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._kagee_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate kagee id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _kagee_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("Kagee id is empty.")
        return text
