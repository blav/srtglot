from dataclasses import dataclass
from functools import cached_property
from typing import Iterable
from bs4 import BeautifulSoup
from datetime import time


@dataclass
class Multiline:
    lines: list[str]

    def __len__(self) -> int:
        return len(self.lines)


@dataclass
class Subtitle:
    start: time
    end: time
    text: list[Multiline]
    soup: BeautifulSoup

    def translate(self, translated_text: list[str]) -> "TranslatedSubtitle":
        lines_count = sum([len(block.lines) for block in self.text])
        if lines_count != len(translated_text):
            raise ValueError(
                f"Number of lines in original and translated text must match. "
                f"Original: {len(self.text)}, Translated: {len(translated_text)}"
            )

        def collect_lines() -> Iterable[str]:
            index = 0
            for block in self.text:
                yield "\n".join(translated_text[index : index + len(block.lines)])
                index += len(block.lines)

        return TranslatedSubtitle(
            start=self.start,
            end=self.end,
            text=self.text,
            soup=self.soup,
            translated_text=[*collect_lines()],
        )

    @cached_property
    def text_lines(self) -> list[str]:
        return [line for block in self.text for line in block.lines]

    def __len__(self) -> int:
        return len(self.text_lines)


@dataclass
class Sentence:
    blocks: list[Subtitle]

    def __str__(self) -> str:
        return " ".join(
            [l for b in self.blocks for m in b.text for l in m.lines if l.strip()]
        )

    def __len__(self) -> int:
        return sum([len(b) for b in self.blocks])


@dataclass
class TranslatedSubtitle(Subtitle):
    translated_text: list[str]

    @cached_property
    def format_translated(self) -> str:
        for t, element in zip(self.translated_text, self.soup.find_all(string=True)):
            element.replace_with(t)

        return str(self.soup)
