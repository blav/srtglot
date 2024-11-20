import re
from typing import Iterable

import openai
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    RetryCallState,
)

from ..model import Sentence, TranslatedSubtitle, TranslatorError
from . import Context


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


def map_to_translated_subtitle(
    batch: list[Sentence], translations: list[str], attempt_number: int
) -> list[TranslatedSubtitle]:
    if len(_to_text(batch)) != len(translations):
        raise TranslatorError(
            f"Attempt {attempt_number}. "
            f"Number of sentences in original and translated text must match. "
            f"Original: {len(_to_text(batch))}, Translated: {len(translations)}"
        )

    completion_iter = iter(translations)
    result = []
    for sentence in batch:
        delimiter = next(completion_iter)
        if not re.match(r"\[sentence \d+\]", delimiter):
            raise TranslatorError(f"delimiter expected, got {delimiter}")

        for sub in sentence.blocks:
            result.append(
                sub.translate(
                    [
                        next(completion_iter) if line.strip() else ""
                        for multiline in sub.text
                        for line in multiline.lines
                    ]
                )
            )

    return result


def translate_batch(
    *,
    context: Context,
    batch: list[Sentence],
) -> list[TranslatedSubtitle]:
    cached = context.cache.get(batch)
    if cached:
        return cached

    def inject_retry_count(retry_state: RetryCallState):
        retry_state.kwargs["attempt_number"] = retry_state.attempt_number

    @retry(
        stop=stop_after_attempt(context.max_attempts),
        before=inject_retry_count,
        retry=retry_if_exception_type(
            (TranslatorError, openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    @context.statistics.register_retry("translate_batch")
    def _translate_batch(attempt_number=None) -> list[TranslatedSubtitle]:
        completion: ChatCompletion = context.client.chat.completions.create(
            model=context.model,
            messages=[
                context.system_message,
                openai.types.chat.ChatCompletionUserMessageParam(
                    role="user",
                    content=_to_prompt_input(batch),
                ),
            ],
        )

        content = completion.choices[0].message.content
        translated_batch = map_to_translated_subtitle(
            batch,
            content.split("\n") if content else [],
            attempt_number,
        )

        context.cache.put(batch, translated_batch)

        return translated_batch

    return _translate_batch()
