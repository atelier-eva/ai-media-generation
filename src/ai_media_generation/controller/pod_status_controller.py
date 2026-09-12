from argparse import ArgumentParser
from sys import argv

from ai_media_generation.infrastructure.runpod import RunPod, write_pod


class PodStatusController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.parse_args(argv[2:])
        write_pod(RunPod().get_pod())
