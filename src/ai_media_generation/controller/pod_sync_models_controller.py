import json
import shlex
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from sys import argv
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh import Ssh
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel
from ai_media_generation.repository.json_io import read_resource_json

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
if dest.is_file():
    print(f"skip {name}", flush=True)
    raise SystemExit(0)
if folder.is_dir():
    for found in folder.rglob(name):
        if not found.is_file():
            continue
        if found.resolve() == dest.resolve():
            continue
        found.replace(dest)
        print(f"moved {found} -> {dest}", flush=True)
        raise SystemExit(0)
os.environ["HF_HOME"] = payload["hf_home"]
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("huggingface_hub is not installed on the pod.", file=sys.stderr)
    raise SystemExit(1)
source = hf_hub_download(
    repo_id=payload["repo"],
    filename=payload["hub_path"],
)
folder.mkdir(parents=True, exist_ok=True)
try:
    os.link(source, dest)
except OSError:
    shutil.copy2(source, dest)
print(f"downloaded {name}", flush=True)
"""


@dataclass(frozen=True)
class SyncModel:
    repo: str
    hub_path: str
    path: str
    profiles: frozenset[str]

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def folder(self) -> str:
        return Path(self.path).parts[1]


def sync_models(profiles: tuple[str, ...] = ()) -> tuple[SyncModel, ...]:
    loaded = read_resource_json("models.json")
    raw = loaded.get("models")
    if not isinstance(raw, list) or not raw:
        raise ValueError("models.json did not list models.")
    models = tuple(_model(item) for item in raw)
    known = frozenset(profile for model in models for profile in model.profiles)
    extra = sorted(set(profiles) - known)
    if extra:
        raise ValueError(f"Unknown profile: {extra[0]}.")
    if not profiles:
        return models
    selected = tuple(
        model for model in models if model.profiles.intersection(profiles)
    )
    if not selected:
        raise ValueError("No models match the given profile(s).")
    return selected


def filenames_by_folder(
    models: tuple[SyncModel, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for model in models:
        grouped.setdefault(model.folder, []).append(model.name)
    return {folder: tuple(names) for folder, names in grouped.items()}


def _model(item: Any) -> SyncModel:
    if not isinstance(item, dict):
        raise ValueError("models.json entry must be an object.")
    repo = _text(item, "repo")
    hub_path = _text(item, "hub_path")
    path = _text(item, "path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid model path: {path}")
    parts = relative.parts
    if len(parts) != 3 or parts[0] != "models" or not parts[1] or not parts[2]:
        raise ValueError(f"Invalid model path: {path}")
    profiles = item.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"models.json profiles missing for {path}.")
    names = frozenset(_profile(value) for value in profiles)
    return SyncModel(repo=repo, hub_path=hub_path, path=path, profiles=names)


def _profile(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("models.json profile must be a non-empty string.")
    return value.strip()


def _text(item: dict[str, Any], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"models.json {name} must be a non-empty string.")
    return value.strip()


class PodSyncModelsController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--profile",
            action="append",
            dest="profiles",
            metavar="NAME",
            help=(
                "Only models for this command (qwen, qwen-edit, "
                "qwen-lora-training, kagee, animagine, animagine-lora-training). "
                "Repeatable. Omit to sync every model."
            ),
        )
        args = parser.parse_args(argv[2:])
        models = sync_models(tuple(args.profiles or ()))
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
                filenames_by_folder(models),
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
