import datetime
import copy
from enum import Enum
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable
from bs4 import BeautifulSoup


class OutputFormat(Enum):
    text = ("text",)
    srt = ("srt",)


@dataclass(frozen=True)
class Multiline:
    lines: list[str]

    def __len__(self) -> int:
        return len(self.lines)


@dataclass(frozen=True)
class Subtitle:
    start: datetime.time
    end: datetime.time
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

        soup = copy.deepcopy(self.soup)
        for t, element in zip(translated_text, soup.find_all(string=True)):
            element.replace_with(t if t else "\n")

        return TranslatedSubtitle.create(
            start=self.start,
            end=self.end,
            text=str(soup),
        )

    @cached_property
    def text_lines(self) -> list[str]:
        return [line for block in self.text for line in block.lines]

    def __len__(self) -> int:
        return len(self.text_lines)


@dataclass(frozen=True)
class Sentence:
    blocks: list[Subtitle]

    def __str__(self) -> str:
        return " ".join(
            [l for b in self.blocks for m in b.text for l in m.lines if l.strip()]
        )

    def __len__(self) -> int:
        return sum([len(b) for b in self.blocks])


@dataclass(frozen=True)
class TranslatedSubtitle:
    start: str
    end: str
    text: str

    @classmethod
    def create(
        cls, start: datetime.time, end: datetime.time, text: str
    ) -> "TranslatedSubtitle":
        return TranslatedSubtitle(
            start=start.strftime("%H:%M:%S,%f")[:-3],
            end=end.strftime("%H:%M:%S,%f")[:-3],
            text=text,
        )
