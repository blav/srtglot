import re
from functools import lru_cache
from typing import Callable, Iterable, Optional
from pathlib import Path

import openai
from openai.types.chat import ChatCompletion
from jinja2 import Template

from .model import Sentence, TranslatedSubtitle, TranslatorError
from .sentence import sentences_batcher
from .cache import Cache


@lru_cache
def _get_system_prompt_template() -> Template:
    return Template((Path(__file__).parent / "prompt.jinja").read_text())


def _get_system_prompt(language: str) -> str:
    return _get_system_prompt_template().render(language=language)


def _to_text(batch: list[Sentence]) -> list[str]:
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

    return [*batch_lines()]


def _to_prompt_input(batch: list[Sentence]) -> str:
    return "\n".join(_to_text(batch))


def _create_openai_client(*, api_key):
    return openai.Client(api_key=api_key)


def translator(
    *,
    model: str,
    language: str,
    max_tokens: int,
    api_key: str,
    cache_dir: Optional[Path] = None,
) -> Callable[[Iterable[Sentence]], Iterable[TranslatedSubtitle]]:
    batcher = sentences_batcher(model, max_tokens)
    cache = Cache(cache_dir=cache_dir)

    system_message = {
        "role": "system",
        "content": _get_system_prompt(language),
    }

    client = _create_openai_client(api_key=api_key)

    def create_translated_subtitle(
        batch: list[Sentence], completion: list[str]
    ) -> Iterable[TranslatedSubtitle]:
        if len(_to_text(batch)) != len(completion):
            raise TranslatorError(
                f"Number of sentences in original and translated text must match. "
                f"Original: {len(batch)}, Translated: {len(completion)}"
            )

        completion_iter = iter(completion)
        for sentence in batch:
            delimiter = next(completion_iter)
            if not re.match(r"\[sentence \d+\]", delimiter):
                raise TranslatorError(f"delimiter expected, got {delimiter}")

            for sub in sentence.blocks:
                translated_text = []
                for multiline in sub.text:
                    for line in multiline.lines:
                        translated_text.append(
                            next(completion_iter) if line.strip() else ""
                        )

                yield sub.translate(translated_text)

    def translate(sentences: Iterable[Sentence]) -> Iterable[TranslatedSubtitle]:
        for batch in batcher(sentences):
            cached = cache.get(batch)
            if cached:
                yield from cached
                continue

            completion: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=[
                    system_message,
                    {
                        "role": "user",
                        "content": _to_prompt_input(batch),
                    },
                ],
            )

            content = completion.choices[0].message.content
            yield from create_translated_subtitle(
                batch, content.split("\n") if content else []
            )

    return translate
