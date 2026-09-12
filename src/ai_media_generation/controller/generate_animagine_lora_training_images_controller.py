from argparse import ArgumentParser
from re import sub
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.animagine.lora_dataset.shoot.generate_shoot_patterns import (
    GenerateShootPatterns,
)
from ai_media_generation.domain.animagine.lora_dataset.shoot.generate_shoot_patterns_output import (
    GenerateShootPatternsOutput,
)
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.animagine_lora_training_generation_log import (
    AnimagineLoraTrainingGenerationLog,
)


class GenerateAnimagineLoraTrainingImagesController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--base-seed",
            type=int,
            default=0,
            help="Seed for --from-row. Later rows increment by 1.",
        )
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument(
            "--from-row",
            type=int,
            default=1,
            help="1-based first prompt row to generate (inclusive).",
        )
        parser.add_argument(
            "--to-row",
            type=int,
            default=0,
            help="1-based last prompt row to generate (inclusive). 0 means the last row.",
        )
        args = parser.parse_args(argv[2:])
        patterns = GenerateShootPatterns().execute()
        start, end = self._row_range(parser, args.from_row, args.to_row, patterns)
        print(f"Processing rows {start + 1}..{end} of {len(patterns)}.")
        config = Config()
        prefix = config.animagine_lora_filename_prefix
        directory = config.animagine_lora_dataset_directory
        comfy_ui = ComfyUi()
        log = AnimagineLoraTrainingGenerationLog()
        for index in range(start, end):
            pattern = patterns[index]
            filename_prefix = self._filename_prefix(pattern, index + 1, prefix)
            fields = self._pattern_fields(pattern)
            seed = args.base_seed + (index - start)
            if log.contains(**fields, seed=seed):
                print(f"[{index + 1}/{end}] {filename_prefix} skip seed={seed}")
                continue
            print(f"[{index + 1}/{end}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_animagine_lora_training_images(
                filename_prefix,
                pattern.width,
                pattern.height,
                pattern.positive_prompt,
                pattern.negative_prompt,
                seed,
                args.batch_size,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
            written = comfy_ui.write_captions(
                images, pattern.caption_prompt, directory
            )
            if written:
                print(f"  captions: {written}")
            log.append(
                **fields,
                row=index + 1,
                seed=seed,
                batch_size=args.batch_size,
                files=self._relative_files(images),
            )
        print(f"Done. {end - start} rows.")

    def _pattern_fields(
        self, pattern: GenerateShootPatternsOutput
    ) -> dict[str, str | None]:
        return {
            "subject": pattern.subject_name,
            "angle": pattern.angle_name,
            "distance": pattern.distance_name,
            "expression": pattern.expression_name,
            "pose": pattern.pose_name,
            "art_style": pattern.art_style_name,
            "background": pattern.background_name,
            "lighting": pattern.lighting_name,
        }

    def _relative_files(
        self, images: tuple[ComfyUi.SavedImage, ...]
    ) -> tuple[str, ...]:
        files: list[str] = []
        for image in images:
            if image.subfolder:
                files.append(f"{image.subfolder}/{image.filename}")
            else:
                files.append(image.filename)
        return tuple(files)

    def _filename_prefix(
        self,
        pattern: GenerateShootPatternsOutput,
        row_number: int,
        prefix: str,
    ) -> str:
        parts = [
            f"{self._slug(prefix)}_{row_number:03d}",
            pattern.subject_name,
            self._slug(pattern.angle_name),
            self._slug(pattern.distance_name),
        ]
        if pattern.expression_name is not None:
            parts.append(self._slug(pattern.expression_name))
        if pattern.pose_name is not None:
            parts.append(self._slug(pattern.pose_name))
        if pattern.art_style_name is not None:
            parts.append(self._slug(pattern.art_style_name))
        if pattern.background_name is not None:
            parts.append(self._slug(pattern.background_name))
        if pattern.lighting_name is not None:
            parts.append(self._slug(pattern.lighting_name))
        return "_".join(parts)

    def _row_range(
        self,
        parser: ArgumentParser,
        from_row: int,
        to_row: int,
        patterns: tuple[GenerateShootPatternsOutput, ...],
    ) -> tuple[int, int]:
        if not patterns:
            raise ValueError("No prompt patterns to generate.")
        if from_row < 1:
            parser.error(
                "--from-row must be 1 or greater. "
                "The first prompt is --from-row 1. "
                "--from-row 0 is not allowed "
                "(unlike --to-row 0, which means the last row)."
            )
        last_row = len(patterns) if to_row <= 0 else to_row
        if last_row > len(patterns):
            parser.error(
                f"--to-row {last_row} is past the last prompt row ({len(patterns)})."
            )
        if from_row > last_row:
            parser.error(
                f"--from-row {from_row} must be <= --to-row {last_row}."
            )
        return from_row - 1, last_row

    def _slug(self, name: str) -> str:
        return sub(r"\s+", "-", name.strip().lower())
