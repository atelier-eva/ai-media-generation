from dataclasses import dataclass

from ai_media_generation.domain.novelai.text.novelai_lorebook import NovelAiLorebook

DEFAULT_MODEL = "llama-3-erato-v1"
DEFAULT_MAX_LENGTH = 100
_JOIN = "\n\n"


@dataclass
class NovelAiTextSpec:
    id: str
    input: str
    model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    system_prompt: str = ""
    memory: str = ""
    lorebooks: tuple[NovelAiLorebook, ...] = ()

    def assembled_input(self) -> str:
        return _JOIN.join(
            part
            for part in (
                self.system_prompt.strip(),
                self.memory.strip(),
                *(lorebook.text.strip() for lorebook in self._active_lorebooks()),
                self.input.strip(),
            )
            if part
        )

    def _active_lorebooks(self) -> tuple[NovelAiLorebook, ...]:
        return tuple(
            lorebook for lorebook in self.lorebooks if lorebook.matches(self.input)
        )
