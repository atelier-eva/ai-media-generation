from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.comfy_ui import ComfyUi
from ai_media_generation.infrastructure.runpod import RunPod, write_pod
from ai_media_generation.infrastructure.ssh_tunnel import SshTunnel


class PodConnectController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.parse_args(argv[2:])
        pod, ssh = RunPod().require_direct_ssh()
        write_pod(pod)
        tunnel = SshTunnel.open(ssh)
        try:
            ComfyUi.wait_until_reachable(
                tunnel.url, Config().runpod_timeout_seconds
            )
            print("Ctrl+C to close the tunnel.")
            tunnel.wait()
        except KeyboardInterrupt:
            print("Tunnel closed.")
        finally:
            tunnel.close()
