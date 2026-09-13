import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_media_generation.infrastructure.error import InfrastructureError


class Ssh:
    _DEFAULT_IDENTITY_NAMES = ("id_ed25519", "id_ecdsa", "id_rsa", "id_dsa")
    _IDENTITY_HINT = (
        "Load the RunPod key with ssh-add, or set RUNPOD_SSH_IDENTITY."
    )

    @dataclass(frozen=True)
    class Endpoint:
        host: str
        port: int
        username: str

    @classmethod
    def argv(
        cls,
        endpoint: "Ssh.Endpoint",
        identity: Path | None,
        extra: tuple[str, ...] = (),
        remote: str | None = None,
    ) -> list[str]:
        ssh = shutil.which("ssh")
        if ssh is None:
            raise InfrastructureError("ssh is not installed.")
        command = [
            ssh,
            "-T",
            *extra,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(endpoint.port),
        ]
        if identity is not None:
            command.extend(["-i", str(identity), "-o", "IdentitiesOnly=yes"])
        command.append(f"{endpoint.username}@{endpoint.host}")
        if remote is not None:
            command.extend(["--", remote])
        return command

    @classmethod
    def require_identities(cls, identity: Path | None) -> None:
        if identity is not None:
            return
        if cls._agent_has_keys() or cls._default_identity_files():
            return
        raise InfrastructureError(
            "No SSH identity is available. " + cls._IDENTITY_HINT
        )

    @classmethod
    def failed(cls, prefix: str, detail: str, code: int | None = None) -> InfrastructureError:
        text = detail.strip() or (
            f"exit {code}" if code is not None else "unknown error"
        )
        if cls._auth_failed(text, code):
            return InfrastructureError(
                f"{prefix}: {text.rstrip('.')}. {cls._IDENTITY_HINT}"
            )
        return InfrastructureError(f"{prefix}: {text}")

    @classmethod
    def run(
        cls,
        endpoint: "Ssh.Endpoint",
        remote: str,
        identity: Path | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> str:
        text = remote.strip()
        if not text:
            raise ValueError("SSH remote command is empty.")
        cls.require_identities(identity)
        argv = cls.argv(endpoint, identity, remote=text)
        try:
            if stream:
                completed = subprocess.run(argv, timeout=timeout)
            else:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(
                f"SSH command timed out after {timeout}s."
            ) from error
        if completed.returncode == 0:
            if stream:
                return ""
            return completed.stdout
        if stream:
            raise cls.failed(
                "SSH command failed",
                f"exit {completed.returncode}",
                completed.returncode,
            )
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit {completed.returncode}"
        )
        raise cls.failed("SSH command failed", detail, completed.returncode)

    @classmethod
    def _agent_has_keys(cls) -> bool:
        ssh_add = shutil.which("ssh-add")
        if ssh_add is None:
            return False
        try:
            completed = subprocess.run(
                [ssh_add, "-l"],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    @classmethod
    def _default_identity_files(cls) -> bool:
        directory = Path.home() / ".ssh"
        return any((directory / name).is_file() for name in cls._DEFAULT_IDENTITY_NAMES)

    @classmethod
    def _auth_failed(cls, detail: str, code: int | None) -> bool:
        lower = detail.lower()
        if "permission denied" in lower or "publickey" in lower:
            return True
        return code == 255 and (
            not detail.strip() or detail.strip() == f"exit {code}"
        )
