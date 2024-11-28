import re
from typing import Iterable

from .model import Sentence
from .context import TranslatorError


def parse_completions(batch: list[Sentence], content: str) -> list[list[str]]:
    lines = [c.strip() for c in content.split("\n") if c.strip()] if content else []

    def is_delimiter(line: str) -> bool:
        return bool(re.match(r"\[sentence \d+\]", line))

    def collect_sentence() -> Iterable[list[str]]:
        sentence: list[str] = None
        for line in lines:
            if sentence is None and not is_delimiter(line):
                raise TranslatorError(batch, lines, "delimiter expected, got {line}")

            if not is_delimiter(line):
                sentence.append(line)
                continue

            if sentence:
                yield sentence

            sentence = []

        yield sentence

    return list(collect_sentence())
