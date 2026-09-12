from dataclasses import dataclass


@dataclass
class Prompt:
    positive_features: tuple[str, ...] = ()
    negative_features: tuple[str, ...] = ()

    def positive_text(self) -> str:
        return self._join(self.positive_features)

    def negative_text(self) -> str:
        return self._join(self.negative_features)

    def _join(self, features: tuple[str, ...]) -> str:
        return ", ".join(name for name in features if name)
