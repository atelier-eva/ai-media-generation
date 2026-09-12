from argparse import ArgumentParser
from sys import argv

from ai_media_generation.config import Config
from ai_media_generation.infrastructure.runpod import RunPod, write_pod


class PodStartController:
    def execute(self, parser: ArgumentParser) -> None:
        parser.parse_args(argv[2:])
        runpod = RunPod()
        print(f"Waiting for RUNNING (timeout {Config().runpod_timeout_seconds}s).")
        write_pod(runpod.start_pod())
