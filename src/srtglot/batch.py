import re

import openai
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    RetryCallState,
)

from .model import Sentence, TranslatedSubtitle
from .translator import Context, TranslatorError
from .prompt import UserPrompt


def _map_to_translated_subtitle(
    prompt: UserPrompt, translations: list[str], attempt_number: int
) -> list[list[TranslatedSubtitle]]:
    if len(prompt.batch_text) != len(translations):
        raise TranslatorError(
            prompt.batch,
            translations,
            f"Attempt {attempt_number}. "
            f"Number of sentences in original and translated text must match. "
            f"Original: {len(prompt.batch_text)}, Translated: {len(translations)}",
        )

    completion_iter = iter(translations)
    result = []
    for sentence in prompt.batch:
        delimiter = next(completion_iter)
        if not re.match(r"\[sentence \d+\]", delimiter):
            raise TranslatorError(
                prompt.batch, translations, f"delimiter expected, got {delimiter}"
            )

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


async def batch_mapper(
    *,
    context: Context,
    batch: list[Sentence],
) -> list[list[TranslatedSubtitle]]:
    cached = await context.cache.get(batch)
    if cached is not None:
        return cached

    def inject_retry_count(retry_state: RetryCallState):
        retry_state.kwargs["attempt_number"] = retry_state.attempt_number

    @retry(
        stop=stop_after_attempt(context.config.max_attempts),
        before=inject_retry_count,
        retry=retry_if_exception_type(
            (openai.APITimeoutError, openai.APIConnectionError)
        ),
    )
    async def _translate_batch(attempt_number=None) -> list[list[TranslatedSubtitle]]:
        prompt = UserPrompt.create_prompt(batch)
        completion: ChatCompletion = await context.client.chat.completions.create(
            model=context.config.model,
            messages=[
                context.system_message,
                prompt.user_message,
            ],
        )

        content = completion.choices[0].message.content
        context.llm_logger(prompt, content)

        translated_batch = _map_to_translated_subtitle(
            prompt,
            [c.strip() for c in content.split("\n") if c.strip()] if content else [],
            attempt_number,
        )

        await context.cache.put(batch, translated_batch)

        return translated_batch

    return await _translate_batch()


async def batch_fallback_mapper(
    sentence: Sentence, exception: TranslatorError
) -> list[TranslatedSubtitle]:
    return [
        TranslatedSubtitle.create(
            start=sub.start,
            end=sub.end,
            text="\n".join(
                [line.strip() for multiline in sub.text for line in multiline.lines]
            ),
        )
        for sub in sentence.blocks
    ]
