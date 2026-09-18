from dataclasses import dataclass
from pathlib import Path


@dataclass
class Subject:
    name: str
    images: tuple[Path, ...]
