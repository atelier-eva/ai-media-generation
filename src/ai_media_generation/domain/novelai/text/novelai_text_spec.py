from dataclasses import dataclass

from ai_media_generation.domain.novelai.text.novelai_lorebook import NovelAiLorebook

DEFAULT_MODEL = "xialong-v1"
DEFAULT_MAX_LENGTH = 300
_JOIN = "\n\n"


@dataclass
class NovelAiTextSpec:
    id: str
    input: str
    model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    stop: tuple[str, ...] = ()
    system_prompt: str = ""
    memory: str = ""
    lorebooks: tuple[NovelAiLorebook, ...] = ()
    output: str = ""
    previous: tuple[str, ...] = ()

    def system_prompt_text(self) -> str:
        return self.system_prompt.strip()

    def context(self) -> str:
        lore = tuple(
            text
            for lorebook in self._active_lorebooks()
            if (text := lorebook.text.strip())
        )
        parts = [part for part in (self.memory.strip(), *lore) if part]
        if lore:
            parts.append("***")
        return _JOIN.join(parts)

    def story(self) -> str:
        return "".join((self.input, *self.previous))

    def _active_lorebooks(self) -> tuple[NovelAiLorebook, ...]:
        return tuple(
            lorebook for lorebook in self.lorebooks if lorebook.matches(self.story())
        )
