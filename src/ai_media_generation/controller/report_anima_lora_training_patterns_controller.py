import csv
from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.domain.anima.lora_dataset.generate_anima_lora_dataset import (
    GenerateAnimaLoraDataset,
)
from ai_media_generation.domain.anima.lora_dataset.generate_anima_lora_dataset_output import (
    AnimaLoraDatasetRow,
)
from ai_media_generation.repository.anima.lora_dataset.shoot_repository import (
    ShootRepository,
)


class ReportAnimaLoraTrainingPatternsController:
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
        rows = GenerateAnimaLoraDataset().execute(ShootRepository().find()).rows
        if not rows:
            raise ValueError("No prompt patterns to generate.")
        directory = Config().anima_lora_dataset_directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "patterns.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self._FIELDNAMES,
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            for row, pattern in enumerate(rows, start=1):
                writer.writerow(self._to_row(row, pattern))
        print(f"Wrote {len(rows)} rows: {path}")

    def _to_row(
        self,
        row: int,
        pattern: AnimaLoraDatasetRow,
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
