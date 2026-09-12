import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.error import InfrastructureError
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel


class RunPod:
    _URL = "https://api.runpod.io/v2"
    _POLL_INTERVAL_SECONDS = 2
    _STARTABLE = frozenset({"EXITED", "ERROR"})
    _STOPPABLE = frozenset({"RUNNING", "PROVISIONING", "STARTING"})
    _STARTING = frozenset({"PROVISIONING", "STARTING"})

    @dataclass(frozen=True)
    class Pod:
        id: str
        name: str
        status: str
        direct_ssh: SshTunnel.Endpoint | None

    def __init__(self) -> None:
        config = Config()
        self._api_key = config.runpod_api_key
        self._pod_id = config.runpod_pod_id
        self._timeout_seconds = config.runpod_timeout_seconds

    def get_pod(self) -> "RunPod.Pod":
        return self._to_pod(self._request("GET", f"/pods/{self._pod_id}"))

    def require_direct_ssh(self) -> tuple["RunPod.Pod", SshTunnel.Endpoint]:
        pod = self.get_pod()
        if pod.status != "RUNNING":
            raise InfrastructureError(
                f"RunPod pod status is {pod.status}. "
                "Run: ai-media-generation pod-start"
            )
        if pod.direct_ssh is None:
            raise InfrastructureError(
                "Direct SSH is unavailable. "
                "Expose 22/tcp on the pod and wait until it is RUNNING."
            )
        return pod, pod.direct_ssh

    def start_pod(self) -> "RunPod.Pod":
        pod = self.get_pod()
        if pod.status == "RUNNING":
            return pod
        if pod.status in self._STARTING:
            return self._wait_until_status("RUNNING")
        if pod.status not in self._STARTABLE:
            raise InfrastructureError(
                f"RunPod pod cannot start from status {pod.status}."
            )
        self._action("start")
        return self._wait_until_status("RUNNING")

    def stop_pod(self) -> "RunPod.Pod":
        pod = self.get_pod()
        if pod.status == "EXITED":
            return pod
        if pod.status not in self._STOPPABLE:
            raise InfrastructureError(
                f"RunPod pod cannot stop from status {pod.status}."
            )
        self._action("stop")
        return self._wait_until_status("EXITED")

    def _action(self, action: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/pods/{self._pod_id}/action", {"action": action}
        )

    def _wait_until_status(self, status: str) -> "RunPod.Pod":
        deadline = time.monotonic() + self._timeout_seconds
        pod: RunPod.Pod | None = None
        while time.monotonic() < deadline:
            pod = self.get_pod()
            if pod.status == status:
                return pod
            if pod.status == "TERMINATED":
                raise InfrastructureError("RunPod pod was terminated.")
            if pod.status == "ERROR" and status != "ERROR":
                raise InfrastructureError(
                    f"RunPod pod entered ERROR while waiting for {status}."
                )
            time.sleep(self._POLL_INTERVAL_SECONDS)
        last = f" (last status {pod.status})" if pod is not None else ""
        raise InfrastructureError(
            f"Timed out after {self._timeout_seconds}s waiting for status "
            f"{status}{last}."
        )

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(
            f"{self._URL}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request) as response:
                loaded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise InfrastructureError(self._http_error_message(path, error)) from error
        except urllib.error.URLError as error:
            raise InfrastructureError(
                f"RunPod is not reachable at {self._URL}"
            ) from error
        except json.JSONDecodeError as error:
            raise InfrastructureError(f"RunPod {path} did not return JSON.") from error
        if not isinstance(loaded, dict):
            raise InfrastructureError(f"RunPod {path} did not return an object.")
        return loaded

    def _http_error_message(self, path: str, error: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            loaded = json.loads(error.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            detail = str(loaded.get("detail") or loaded.get("title") or "").strip()
        if detail:
            return f"RunPod {path} failed: {error.code} {detail}"
        return f"RunPod {path} failed: {error.code}"

    def _to_pod(self, data: dict[str, Any]) -> "RunPod.Pod":
        identifier = str(data.get("id") or "").strip()
        if not identifier:
            raise InfrastructureError("RunPod pod did not return id.")
        name = str(data.get("name") or "").strip()
        if not name:
            raise InfrastructureError("RunPod pod did not return name.")
        status = str(data.get("status") or "").strip()
        if not status:
            raise InfrastructureError("RunPod pod did not return status.")
        return RunPod.Pod(
            id=identifier,
            name=name,
            status=status,
            direct_ssh=self._direct_ssh(data.get("ssh")),
        )

    def _direct_ssh(self, value: Any) -> SshTunnel.Endpoint | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise InfrastructureError("RunPod pod ssh was not an object.")
        raw = value.get("direct")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise InfrastructureError("RunPod pod ssh.direct was not an object.")
        host = str(raw.get("host") or "").strip()
        username = str(raw.get("username") or "").strip()
        port = self._port(raw.get("port"))
        if not host or not username or port is None:
            raise InfrastructureError("RunPod pod ssh.direct is incomplete.")
        return SshTunnel.Endpoint(host=host, port=port, username=username)

    def _port(self, value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                port = int(text)
            except ValueError:
                return None
            return port if port > 0 else None
        return None


def write_pod(pod: RunPod.Pod) -> None:
    print(f"id: {pod.id}")
    print(f"name: {pod.name}")
    print(f"status: {pod.status}")
    if pod.direct_ssh is None:
        print("direct ssh: unavailable")
        return
    ssh = pod.direct_ssh
    print(f"direct ssh: {ssh.username}@{ssh.host}:{ssh.port}")
