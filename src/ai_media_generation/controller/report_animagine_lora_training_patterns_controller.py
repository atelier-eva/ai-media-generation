import csv
from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.shoot.generate_shoot_patterns import (
    GenerateShootPatterns,
)
from ai_media_generation.domain.shoot.generate_shoot_patterns_output import (
    GenerateShootPatternsOutput,
)


class ReportAnimagineLoraTrainingPatternsController:
    _FIELDNAMES = (
        "row",
        "subject",
        "angle",
        "distance",
        "expression",
        "pose",
        "art_style",
        "background",
        "lighting",
        "width",
        "height",
    )

    def execute(self, parser: ArgumentParser) -> None:
        parser.parse_args(argv[2:])
        patterns = GenerateShootPatterns().execute()
        if not patterns:
            raise ValueError("No prompt patterns to generate.")
        directory = Config().animagine_lora_dataset_directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "patterns.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self._FIELDNAMES,
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            for row, pattern in enumerate(patterns, start=1):
                writer.writerow(self._to_row(row, pattern))
        print(f"Wrote {len(patterns)} rows: {path}")

    def _to_row(
        self,
        row: int,
        pattern: GenerateShootPatternsOutput,
    ) -> dict[str, str | int]:
        return {
            "row": row,
            "subject": pattern.subject_name,
            "angle": pattern.angle_name,
            "distance": pattern.distance_name,
            "expression": pattern.expression_name or "",
            "pose": pattern.pose_name or "",
            "art_style": pattern.art_style_name or "",
            "background": pattern.background_name or "",
            "lighting": pattern.lighting_name or "",
            "width": pattern.width,
            "height": pattern.height,
        }
