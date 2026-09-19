import json
import shlex
from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh import Ssh
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel
from ai_media_generation.repository.model_repository import ModelRepository, SyncModel

_HF_HOME = "/workspace/hf"
_REMOTE_SCRIPT = """
import os
import shutil
import sys
from pathlib import Path

payload = __PAYLOAD__
dest = Path(payload["root"]) / payload["path"]
name = dest.name
folder = dest.parent

def usable(path):
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0

if dest.is_symlink() or (dest.exists() and not usable(dest)):
    dest.unlink()
if usable(dest):
    print(f"skip {name}", flush=True)
    raise SystemExit(0)
if folder.is_dir():
    for found in folder.rglob(name):
        if not usable(found):
            continue
        if found.resolve() == dest.resolve():
            continue
        found.replace(dest)
        if not usable(dest):
            raise SystemExit(f"dest is not a usable regular file: {dest}")
        print(f"moved {found} -> {dest}", flush=True)
        raise SystemExit(0)
os.environ["HF_HOME"] = payload["hf_home"]
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub is not installed on the pod.", file=sys.stderr)
    raise SystemExit(1)
source = Path(
    hf_hub_download(
        repo_id=payload["repo"],
        filename=payload["hub_path"],
    )
).resolve()
if dest.is_symlink() or dest.exists():
    dest.unlink()
folder.mkdir(parents=True, exist_ok=True)
try:
    os.link(source, dest)
except OSError:
    shutil.copy2(source, dest, follow_symlinks=True)
if not usable(dest):
    raise SystemExit(f"dest is not a usable regular file: {dest}")
print(f"downloaded {name}", flush=True)
"""


class PodSyncModelsController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--profile",
            action="append",
            dest="profiles",
            metavar="NAME",
            help=(
                "Only models for this command (qwen, qwen-edit, "
                "qwen-lora-training, anima, anima-lora-training, kagee, "
                "animagine, animagine-lora-training). "
                "Repeatable. Omit to sync every model."
            ),
        )
        args = parser.parse_args(argv[2:])
        repository = ModelRepository()
        models = repository.get(tuple(args.profiles or ()))
        pod, ssh = RunPod().require_direct_ssh()
        write_pod(pod)
        config = Config()
        print(f"Syncing {len(models)} model(s) to {config.runpod_comfyui_root}.")
        for index, model in enumerate(models, start=1):
            print(f"[{index}/{len(models)}] {model.name}")
            Ssh.run(
                ssh,
                self._remote(config.runpod_comfyui_root, model),
                identity=config.runpod_ssh_identity,
                stream=True,
            )
        tunnel = SshTunnel.open(ssh)
        try:
            ComfyUi.wait_until_reachable(
                tunnel.url, config.runpod_timeout_seconds
            )
            ComfyUi.require_filenames(
                tunnel.url,
                repository.filenames_by_folder(models),
                "It must appear as a top-level filename, not under a subdirectory.",
            )
        finally:
            tunnel.close()
        print(f"Done. {len(models)} model(s).")

    def _remote(self, root: str, model: SyncModel) -> str:
        payload = {
            "root": root,
            "repo": model.repo,
            "hub_path": model.hub_path,
            "path": model.path,
            "hf_home": _HF_HOME,
        }
        script = _REMOTE_SCRIPT.replace("__PAYLOAD__", json.dumps(payload), 1)
        return "python3 -c " + shlex.quote(script)
