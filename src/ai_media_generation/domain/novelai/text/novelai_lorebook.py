from dataclasses import dataclass


@dataclass
class NovelAiLorebook:
    id: str
    text: str
    keys: tuple[str, ...] = ()

    def activation_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        seen: set[str] = set()
        for key in self.keys:
            folded = key.casefold()
            if not folded or folded in seen:
                continue
            seen.add(folded)
            keys.append(key)
        return tuple(keys)

    def matches(self, story: str) -> bool:
        haystack = story.casefold()
        return any(key.casefold() in haystack for key in self.activation_keys())
