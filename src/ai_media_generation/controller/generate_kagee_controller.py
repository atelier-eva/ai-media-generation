from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.controller.helper import add_remote_arguments, require_remote_models
from ai_media_generation.domain.kagee_spec.get_kagee_specs import GetKageeSpecs
from ai_media_generation.domain.kagee_spec.get_kagee_specs_output import KageeSpecDto
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel


class GenerateKageeController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        add_remote_arguments(parser)
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
        tunnel: SshTunnel | None = None
        try:
            if args.remote:
                pod, ssh = RunPod().require_direct_ssh()
                write_pod(pod)
                tunnel = SshTunnel.open(ssh)
                ComfyUi.wait_until_reachable(
                    tunnel.url, Config().runpod_timeout_seconds
                )
                require_remote_models(tunnel.url, "kagee")
            url = tunnel.url if tunnel is not None else Config().comfy_ui_url
            self._generate(specs, args.base_seed, url)
        finally:
            if tunnel is not None:
                tunnel.close()

    def _generate(
        self,
        specs: tuple[KageeSpecDto, ...],
        base_seed: int,
        url: str,
    ) -> None:
        directory = Config().kagee_output_directory
        comfy_ui = ComfyUi(url)
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = spec.seed if spec.seed is not None else base_seed + index
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
