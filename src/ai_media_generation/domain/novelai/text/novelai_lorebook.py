from dataclasses import dataclass
from pathlib import Path


@dataclass
class NovelAiLorebook:
    id: str
    text: str
    keys: tuple[str, ...] = ()

    def activation_keys(self) -> tuple[str, ...]:
        stem = Path(self.id).name
        keys = [stem]
        seen = {stem.casefold()}
        for key in self.keys:
            folded = key.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            keys.append(key)
        return tuple(keys)

    def matches(self, story: str) -> bool:
        haystack = story.casefold()
        return any(key.casefold() in haystack for key in self.activation_keys())
