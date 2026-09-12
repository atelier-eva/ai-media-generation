import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.error import InfrastructureError


class SshTunnel:
    _LOCAL_HOST = "127.0.0.1"
    _LOCAL_PORT = 8188
    _REMOTE_HOST = "127.0.0.1"
    _REMOTE_PORT = 8188
    _LISTEN_TIMEOUT_SECONDS = 30
    _POLL_INTERVAL_SECONDS = 0.2

    @dataclass(frozen=True)
    class Endpoint:
        host: str
        port: int
        username: str

    def __init__(
        self, ssh: "SshTunnel.Endpoint", identity: Path | None = None
    ) -> None:
        self._ssh = ssh
        self._identity = identity
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr = ""
        self._stderr_thread: Thread | None = None

    @classmethod
    def local_url(cls) -> str:
        return f"http://{cls._LOCAL_HOST}:{cls._LOCAL_PORT}"

    @classmethod
    def open(cls, ssh: "SshTunnel.Endpoint") -> "SshTunnel":
        tunnel = cls(ssh, Config().runpod_ssh_identity)
        print(f"Forwarding {tunnel.url} -> 127.0.0.1:8188")
        try:
            tunnel.start()
            tunnel.wait_until_listening()
        except BaseException:
            tunnel.close()
            raise
        return tunnel

    @property
    def url(self) -> str:
        return self.local_url()

    def start(self) -> None:
        if self._process is not None:
            raise InfrastructureError("SSH tunnel is already started.")
        ssh = shutil.which("ssh")
        if ssh is None:
            raise InfrastructureError("ssh is not installed.")
        self._process = subprocess.Popen(
            self._command(ssh),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._stderr_thread = Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def wait_until_listening(self) -> None:
        deadline = time.monotonic() + self._LISTEN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            self._raise_if_exited()
            try:
                with socket.create_connection(
                    (self._LOCAL_HOST, self._LOCAL_PORT), timeout=1
                ):
                    return
            except OSError:
                time.sleep(self._POLL_INTERVAL_SECONDS)
        raise InfrastructureError(
            f"Timed out after {self._LISTEN_TIMEOUT_SECONDS}s waiting for "
            f"SSH tunnel on {self._LOCAL_HOST}:{self._LOCAL_PORT}."
        )

    def wait(self) -> None:
        if self._process is None:
            raise InfrastructureError("SSH tunnel is not started.")
        self._process.wait()
        self._wait_for_stderr()
        detail = self._stderr.strip() or f"exit {self._process.returncode}"
        raise InfrastructureError(f"SSH tunnel exited: {detail}")

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._wait_for_stderr()

    def _command(self, ssh: str) -> list[str]:
        command = [
            ssh,
            "-N",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
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
            "-L",
            (
                f"{self._LOCAL_HOST}:{self._LOCAL_PORT}:"
                f"{self._REMOTE_HOST}:{self._REMOTE_PORT}"
            ),
            "-p",
            str(self._ssh.port),
        ]
        if self._identity is not None:
            command.extend(
                ["-i", str(self._identity), "-o", "IdentitiesOnly=yes"]
            )
        command.append(f"{self._ssh.username}@{self._ssh.host}")
        return command

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        self._stderr = process.stderr.read().decode("utf-8", errors="replace")

    def _wait_for_stderr(self) -> None:
        thread = self._stderr_thread
        if thread is not None:
            thread.join(timeout=1)

    def _raise_if_exited(self) -> None:
        process = self._process
        if process is None:
            raise InfrastructureError("SSH tunnel is not started.")
        code = process.poll()
        if code is None:
            return
        self._wait_for_stderr()
        detail = self._stderr.strip() or f"exit {code}"
        raise InfrastructureError(f"SSH tunnel exited: {detail}")
