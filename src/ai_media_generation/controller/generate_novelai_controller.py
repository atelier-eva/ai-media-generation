from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.novelai.spec.get_novelai_specs import GetNovelAiSpecs
from ai_media_generation.infrastructure.novelai import NovelAI


class GenerateNovelAiController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "NovelAI JSON paths under novelai/spec/, relative, nested allowed "
                "(e.g. hero/smile.json). Omit to generate every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetNovelAiSpecs().execute(self._novelai_ids(args.files)).dtos
        if not specs:
            raise ValueError("No novelai JSON to generate.")
        print(f"Processing {len(specs)} novelai JSON file(s).")
        config = Config()
        directory = config.novelai_output_directory
        novelai = NovelAI()
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = args.base_seed + index
            print(f"[{index + 1}/{len(specs)}] {filename_prefix} seed={seed}")
            images = novelai.generate_image(
                filename_prefix,
                spec.width,
                spec.height,
                spec.positive_prompt,
                spec.negative_prompt,
                seed,
                spec.model,
                spec.sampler,
                spec.steps,
                spec.scale,
                args.batch_size,
                tuple(
                    NovelAI.CharacterPrompt(
                        positive=character.positive_prompt,
                        negative=character.negative_prompt,
                        x=character.x,
                        y=character.y,
                    )
                    for character in spec.characters
                ),
                spec.use_order,
                None
                if spec.img2img is None
                else NovelAI.Img2Img(
                    image=spec.img2img.image,
                    strength=spec.img2img.strength,
                    noise=spec.img2img.noise,
                ),
            )
            written = novelai.write_images(images, directory)
            if written:
                print(f"  images: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _novelai_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._novelai_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate novelai id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _novelai_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("NovelAI id is empty.")
        return text
