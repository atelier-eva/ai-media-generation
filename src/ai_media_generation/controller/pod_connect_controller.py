from argparse import ArgumentParser
from sys import argv

from ai_media_generation.infrastructure.error import InfrastructureError
from ai_media_generation.infrastructure.remote_session import open_ssh_tunnel
from ai_media_generation.infrastructure.runpod import RunPod, write_pod


class PodConnectController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.parse_args(argv[2:])
        pod = RunPod().get_pod()
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
        write_pod(pod)
        tunnel = open_ssh_tunnel(pod.direct_ssh)
        try:
            print("Ctrl+C to close the tunnel.")
            tunnel.wait()
        except KeyboardInterrupt:
            print("Tunnel closed.")
        finally:
            tunnel.close()
