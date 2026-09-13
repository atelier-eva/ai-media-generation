import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_media_generation.infrastructure.error import InfrastructureError


class Ssh:
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
            raise InfrastructureError(
                f"SSH command failed: exit {completed.returncode}"
            )
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit {completed.returncode}"
        )
        raise InfrastructureError(f"SSH command failed: {detail}")
