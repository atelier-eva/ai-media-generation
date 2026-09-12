from argparse import ArgumentParser


def add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "Forward localhost:8188 over Direct SSH to the running RunPod pod, "
            "and generate through that tunnel."
        ),
    )
