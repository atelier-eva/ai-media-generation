from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.controller.helper import add_remote_arguments, require_remote_models
from ai_media_generation.domain.anima.spec.get_anima_specs import GetAnimaSpecs
from ai_media_generation.domain.anima.spec.get_anima_specs_output import AnimaSpecDto
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel


class GenerateAnimaController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument("--base-seed", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=4)
        add_remote_arguments(parser)
        parser.add_argument(
            "files",
            nargs="*",
            help=(
                "Anima JSON paths under anima/spec/, relative, nested allowed "
                "(e.g. hero/smile.json). Omit to generate every file."
            ),
        )
        args = parser.parse_args(argv[2:])
        specs = GetAnimaSpecs().execute(self._anima_ids(args.files)).dtos
        if not specs:
            raise ValueError("No anima JSON to generate.")
        print(f"Processing {len(specs)} anima JSON file(s).")
        tunnel: SshTunnel | None = None
        try:
            if args.remote:
                pod, ssh = RunPod().require_direct_ssh()
                write_pod(pod)
                tunnel = SshTunnel.open(ssh)
                ComfyUi.wait_until_reachable(
                    tunnel.url, Config().runpod_timeout_seconds
                )
                require_remote_models(tunnel.url, "anima")
            url = tunnel.url if tunnel is not None else Config().comfy_ui_url
            self._generate(specs, args.base_seed, args.batch_size, url)
        finally:
            if tunnel is not None:
                tunnel.close()

    def _generate(
        self,
        specs: tuple[AnimaSpecDto, ...],
        base_seed: int,
        batch_size: int,
        url: str,
    ) -> None:
        config = Config()
        directory = config.anima_output_directory
        comfy_ui = ComfyUi(url)
        for index, spec in enumerate(specs):
            filename_prefix = spec.id
            seed = base_seed + index
            print(f"[{index + 1}/{len(specs)}] {filename_prefix} seed={seed}")
            images = comfy_ui.generate_anima(
                filename_prefix,
                spec.width,
                spec.height,
                spec.prompt,
                spec.negative,
                seed,
                batch_size,
            )
            written = comfy_ui.write_images(images, directory)
            if written:
                print(f"  images: {written}")
        print(f"Done. {len(specs)} file(s).")

    def _anima_ids(self, files: list[str]) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for raw in files:
            identifier = self._anima_id(raw)
            if identifier in seen:
                raise ValueError(f"Duplicate anima id: {identifier}")
            seen.add(identifier)
            ids.append(identifier)
        return tuple(ids)

    def _anima_id(self, value: str) -> str:
        text = value.strip().replace("\\", "/")
        if text.endswith(".json"):
            text = text[: -len(".json")]
        text = text.strip("/")
        if not text:
            raise ValueError("Anima id is empty.")
        return text
