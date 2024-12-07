import re
from typing import Iterable

from srtglot.prompt import UserPrompt

from .model import Sentence, TranslatedSubtitle
from .context import TranslatorError


def parse_completions(batch: list[Sentence], content: str) -> list[list[str]]:
    lines = [c.strip() for c in content.split("\n") if c.strip()] if content else []

    def is_delimiter(line: str) -> bool:
        return bool(re.match(r"\[sentence \d+\]", line))

    def collect_sentence() -> Iterable[list[str]]:
        init: list[str] = []
        sentence: list[str] = init
        for line in lines:
            if sentence is init and not is_delimiter(line):
                raise TranslatorError(batch, lines, f"Delimiter expected, got {line}")

            if not is_delimiter(line):
                sentence.append(line)
                continue

            if sentence:
                yield sentence

            sentence = []

        yield sentence

    completions = list(collect_sentence())
    if len(completions) != len(batch):
        raise TranslatorError(
            batch,
            lines,
            f"Number of sentences in prompt and completions must match. "
            f"Prompt: {len(batch)}, Completions: {len(completions)}",
        )

    for sentence, completion in zip(batch, completions):
        if abs(sentence.non_empty_text_lines_count - len(completion)) > 2:
            raise TranslatorError(
                batch,
                lines,
                f"Number of sentence fragments differ by more than 2 from completions. "
                f"Prompt: {sentence.non_empty_text_lines_count}, Completions: {len(completion)}",
            )

        if len(completion) == 0:
            raise TranslatorError(batch, completion, "Empty completion")

    return completions


def map_to_translated_subtitle(
    sentences: list[Sentence], parsed_completions: list[list[str]], attempt_number: int
) -> list[list[TranslatedSubtitle]]:
    result = []
    for sentence, completions in zip(sentences, parsed_completions):
        completion_iter = iter(completions)
        result.append(
            [
                sub.translate(
                    [
                        next(completion_iter) if line.strip() else ""
                        for multiline in sub.text
                        for line in multiline.lines
                    ]
                )
                for sub in sentence.blocks
            ]
        )

    return result
