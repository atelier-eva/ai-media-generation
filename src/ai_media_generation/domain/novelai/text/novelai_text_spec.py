from dataclasses import dataclass

from ai_media_generation.domain.novelai.text.novelai_lorebook import NovelAiLorebook

DEFAULT_MODEL = "xialong-v1"
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
    output: str = ""
    previous: tuple[str, ...] = ()

    def system_prompt_text(self) -> str:
        return self.system_prompt.strip()

    def assembled_input(self) -> str:
        return _JOIN.join(
            part
            for part in (
                self.memory.strip(),
                *(lorebook.text.strip() for lorebook in self._active_lorebooks()),
                self.story(),
            )
            if part
        )

    def story(self) -> str:
        return "".join((self.input.strip(), *self.previous))

    def _active_lorebooks(self) -> tuple[NovelAiLorebook, ...]:
        return tuple(
            lorebook for lorebook in self.lorebooks if lorebook.matches(self.story())
        )
