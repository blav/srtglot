from typing import Any
from collections.abc import Callable, Coroutine


from .model import Sentence, TranslatedSubtitle
from .context import Context, TranslatorError
from .adaptive import adaptive_map


def translator(
    context: Context,
) -> Callable[[list[Sentence]], Coroutine[Any, Any, list[list[TranslatedSubtitle]]]]:
    async def translate(sentences: list[Sentence]) -> list[list[TranslatedSubtitle]]:
        from .batch import batch_mapper, batch_fallback_mapper

        async def mapper(batch: list[Sentence]) -> list[list[TranslatedSubtitle]]:
            return await batch_mapper(
                context=context,
                batch=batch,
            )
        
        async def fallback_mapper(sentence: Sentence, exception: TranslatorError) -> list[TranslatedSubtitle]:
            return await batch_fallback_mapper(
                context=context,
                sentence=sentence,
                exception=exception,
            )

        return await adaptive_map(
            sentences,
            mapper,
            fallback_mapper,
            TranslatorError,
        )

    return translate
