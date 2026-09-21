from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.controller.helper import add_remote_arguments, require_remote_models
from ai_media_generation.domain.qwen.edit.get_qwen_edit_specs import GetQwenEditSpecs
from ai_media_generation.domain.qwen.edit.get_qwen_edit_specs_output import (
    QwenEditSpecDto,
)
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel


class GenerateQwenEditController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        add_remote_arguments(parser)
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "Qwen edit JSON paths under qwen/edit/spec/, relative, nested "
                "allowed (e.g. hero/smile.json). Omit to generate every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetQwenEditSpecs().execute(self._qwen_edit_ids(args.files)).dtos
        if not specs:
            raise ValueError("No qwen edit JSON to generate.")
        print(f"Processing {len(specs)} qwen edit JSON file(s).")
        tunnel: SshTunnel | None = None
        try:
            if args.remote:
                pod, ssh = RunPod().require_direct_ssh()
                write_pod(pod)
                tunnel = SshTunnel.open(ssh)
                ComfyUi.wait_until_reachable(
                    tunnel.url, Config().runpod_timeout_seconds
                )
                require_remote_models(tunnel.url, "qwen-edit")
            url = tunnel.url if tunnel is not None else Config().comfy_ui_url
            self._generate(specs, args.base_seed, url)
        finally:
            if tunnel is not None:
                tunnel.close()

    def _generate(
        self,
        specs: tuple[QwenEditSpecDto, ...],
        base_seed: int,
        url: str,
    ) -> None:
        config = Config()
        directory = config.qwen_edit_output_directory
        comfy_ui = ComfyUi(url)
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = spec.seed if spec.seed is not None else base_seed + index
            print(f"[{index + 1}/{len(specs)}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_qwen_edit(
                filename_prefix,
                spec.images,
                spec.prompt,
                seed,
                spec.negative,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _qwen_edit_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._qwen_edit_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate qwen edit id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _qwen_edit_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("Qwen edit id is empty.")
        return text
