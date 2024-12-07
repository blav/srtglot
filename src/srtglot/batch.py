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
from .completions import map_to_translated_subtitle, parse_completions
from .fallback import fit_fragments_count


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

        parsed_completions = parse_completions(batch, content or "")
        tokenization = context.config.target_language.value.tokenization
        parsed_completions = [
            fit_fragments_count(
                tokenization,
                sentence.non_empty_text_lines_count,
                parsed_completion,
            )
            for sentence, parsed_completion in zip(batch, parsed_completions)
        ]

        translated_batch = map_to_translated_subtitle(
            batch, parsed_completions, attempt_number
        )

        await context.cache.put(batch, translated_batch)

        return translated_batch

    return await _translate_batch()


async def batch_fallback_mapper(
    *, context: Context, sentence: Sentence, exception: TranslatorError
) -> list[TranslatedSubtitle]:
    return (await batch_mapper(context=context, batch=[sentence]))[0]
