from argparse import ArgumentParser
from re import sub
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.controller.helper import add_remote_arguments, require_remote_models
from ai_media_generation.domain.qwen.lora_dataset.generate_qwen_lora_dataset import (
    GenerateQwenLoraDataset,
)
from ai_media_generation.domain.qwen.lora_dataset.generate_qwen_lora_dataset_output import (
    QwenLoraDatasetRow,
)
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.qwen_lora_training_generation_log import (
    QwenLoraTrainingGenerationLog,
)
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel
from ai_media_generation.repository.qwen.lora_dataset.shoot_repository import (
    ShootRepository,
)


class GenerateQwenLoraTrainingImagesController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--base-seed",
            type=int,
            default=0,
            help="Seed for --from-row. Later rows increment by 1.",
        )
        parser.add_argument(
            "--from-row",
            type=int,
            default=1,
            help="1-based first dataset row to generate (inclusive).",
        )
        parser.add_argument(
            "--to-row",
            type=int,
            default=0,
            help="1-based last dataset row to generate (inclusive). 0 means the last row.",
        )
        add_remote_arguments(parser)
        args = parser.parse_args(argv[2:])
        rows = GenerateQwenLoraDataset().execute(ShootRepository().find()).rows
        start, end = self._row_range(parser, args.from_row, args.to_row, rows)
        print(f"Processing rows {start + 1}..{end} of {len(rows)}.")
        tunnel: SshTunnel | None = None
        try:
            if args.remote:
                pod, ssh = RunPod().require_direct_ssh()
                write_pod(pod)
                tunnel = SshTunnel.open(ssh)
                ComfyUi.wait_until_reachable(
                    tunnel.url, Config().runpod_timeout_seconds
                )
                require_remote_models(tunnel.url, "qwen-lora-training")
            url = tunnel.url if tunnel is not None else Config().comfy_ui_url
            self._generate(rows, start, end, args.base_seed, url)
        finally:
            if tunnel is not None:
                tunnel.close()

    def _generate(
        self,
        rows: tuple[QwenLoraDatasetRow, ...],
        start: int,
        end: int,
        base_seed: int,
        url: str,
    ) -> None:
        config = Config()
        prefix = config.qwen_lora_filename_prefix
        directory = config.qwen_lora_dataset_directory
        comfy_ui = ComfyUi(url)
        log = QwenLoraTrainingGenerationLog()
        for index in range(start, end):
            row = rows[index]
            filename_prefix = self._filename_prefix(row, index + 1, prefix)
            fields = self._row_fields(row)
            seed = base_seed + (index - start)
            if log.contains(**fields, seed=seed):
                print(f"[{index + 1}/{end}] {filename_prefix} skip seed={seed}")
                continue
            print(f"[{index + 1}/{end}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_qwen_lora_training_images(
                filename_prefix,
                row.images,
                row.edit_prompt,
                seed,
                row.negative_prompt,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
            written = comfy_ui.write_captions(images, row.caption_prompt, directory)
            if written:
                print(f"  captions: {written}")
            log.append(
                **fields,
                row=index + 1,
                seed=seed,
                files=self._relative_files(images),
            )
        print(f"Done. {end - start} rows.")

    def _row_fields(self, row: QwenLoraDatasetRow) -> dict[str, str | None]:
        return {
            "subject": row.subject_name,
            "angle": row.angle_name,
            "distance": row.distance_name,
            "expression": row.expression_name,
            "pose": row.pose_name,
            "art_style": row.art_style_name,
            "background": row.background_name,
            "lighting": row.lighting_name,
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
        row: QwenLoraDatasetRow,
        row_number: int,
        prefix: str,
    ) -> str:
        parts = [
            f"{self._slug(prefix)}_{row_number:03d}",
            self._slug(row.subject_name),
            self._slug(row.angle_name),
            self._slug(row.distance_name),
        ]
        if row.expression_name is not None:
            parts.append(self._slug(row.expression_name))
        if row.pose_name is not None:
            parts.append(self._slug(row.pose_name))
        if row.art_style_name is not None:
            parts.append(self._slug(row.art_style_name))
        if row.background_name is not None:
            parts.append(self._slug(row.background_name))
        if row.lighting_name is not None:
            parts.append(self._slug(row.lighting_name))
        return "_".join(parts)

    def _row_range(
        self,
        parser: ArgumentParser,
        from_row: int,
        to_row: int,
        rows: tuple[QwenLoraDatasetRow, ...],
    ) -> tuple[int, int]:
        if not rows:
            raise ValueError("No Qwen LoRA dataset rows to generate.")
        if from_row < 1:
            parser.error(
                "--from-row must be 1 or greater. "
                "The first row is --from-row 1. "
                "--from-row 0 is not allowed "
                "(unlike --to-row 0, which means the last row)."
            )
        last_row = len(rows) if to_row <= 0 else to_row
        if last_row > len(rows):
            parser.error(
                f"--to-row {last_row} is past the last dataset row ({len(rows)})."
            )
        if from_row > last_row:
            parser.error(
                f"--from-row {from_row} must be <= --to-row {last_row}."
            )
        return from_row - 1, last_row

    def _slug(self, name: str) -> str:
        return sub(r"\s+", "-", name.strip().lower())
