from functools import lru_cache
from typing import Callable, Iterable
import openai
from jinja2 import Template
from pathlib import Path
from .model import Sentence, TranslatedSubtitle, Subtitle
from .sentence import sentences_batcher


@lru_cache
def _get_system_prompt_template() -> Template:
    return Template((Path(__file__).parent / "prompt.jinja").read_text())


def _get_system_prompt(language: str) -> str:
    return _get_system_prompt_template().render(language=language)


def _to_text(batch: list[Sentence]) -> list[str]:
    return [
        line
        for sentence in batch
        for sub in sentence.blocks
        for multiline in sub.text
        for line in multiline.lines
        if line.strip()
    ]


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
) -> Callable[[Iterable[Sentence]], Iterable[TranslatedSubtitle]]:
    batcher = sentences_batcher(model, max_tokens)

    system_message = {
        "role": "system",
        "content": _get_system_prompt(language),
    }

    client = _create_openai_client(api_key=api_key)

    def create_translated_subtitle(
        batch: list[Sentence], completion: list[str]
    ) -> Iterable[TranslatedSubtitle]:
        if len(_to_text(batch)) != len(completion):
            raise ValueError(
                f"Number of sentences in original and translated text must match. "
                f"Original: {len(batch)}, Translated: {len(completion)}"
            )

        def subtitles() -> Iterable[Subtitle]:
            for sentence in batch:
                yield from sentence.blocks

        translations = iter(completion)
        for subtitle in subtitles():
            translated_text = []
            for multiline in subtitle.text:
                for line in multiline.lines:
                    translated_text.append(next(translations) if line else "")

            yield subtitle.translate(translated_text)

    def translate(sentences: Iterable[Sentence]) -> Iterable[TranslatedSubtitle]:
        for batch in batcher(sentences):
            completion: openai.types.Completion = client.completions.create(
                model=model,
                messages=[
                    system_message,
                    {"role": "user", "content": _to_prompt_input(batch)},
                ],
            )

            yield from create_translated_subtitle(
                batch, completion.choices[0].text.split("\n")
            )

    return translate
