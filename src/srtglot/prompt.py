from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterable

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from jinja2 import Template

from .model import Sentence
from .config import Config


@lru_cache
def _get_system_prompt_template() -> Template:
    return Template((Path(__file__).parent / "prompt.jinja").read_text())


def get_system_prompt(config: Config) -> ChatCompletionSystemMessageParam:
    language = config.target_language
    content = _get_system_prompt_template().render(language=language.value.name)
    return ChatCompletionSystemMessageParam(
        role="system",
        content=content,
    )


@dataclass(frozen=True)
class UserPrompt:
    batch: list[Sentence]
    batch_text: list[str]

    def __len__(self) -> int:
        return len(self.batch_text)

    @cached_property
    def user_message(self) -> ChatCompletionUserMessageParam:
        return ChatCompletionUserMessageParam(
            role="user",
            content="\n".join(self.batch_text),
        )

    @classmethod
    def create_prompt(cls, batch: list[Sentence]) -> "UserPrompt":
        def lines(sentence: Sentence) -> Iterable[str]:
            for block in sentence.blocks:
                for multiline in block.text:
                    for line in multiline.lines:
                        if line.strip():
                            yield line

        def batch_lines() -> Iterable[str]:
            for i, sentence in enumerate(batch):
                yield f"[sentence {i + 1}]"
                yield from lines(sentence)

        return cls(batch, [*batch_lines()])
