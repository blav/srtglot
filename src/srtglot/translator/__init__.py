from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, List, Optional, AsyncGenerator
from pathlib import Path
from itertools import islice
from logging import getLogger, Logger, FileHandler, NullHandler
import asyncio

import openai
from jinja2 import Template

from ..model import Sentence, TranslatedSubtitle
from ..sentence import sentences_batcher, Batcher
from ..cache import Cache
from ..languages import Language
from ..statistics import Statistics
from .adaptive import adaptive_map


class TranslatorError(ValueError):
    batch: list[Sentence]
    completions: list[str]

    def __init__(self, batch: List[Sentence], completions: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)


@lru_cache
def _get_system_prompt_template() -> Template:
    return Template((Path(__file__).parent / "prompt.jinja").read_text())


def _get_system_prompt(language: Language) -> str:
    return _get_system_prompt_template().render(language=language.value)


def _create_openai_client(*, api_key) -> openai.AsyncClient:
    return openai.AsyncClient(api_key=api_key)


@dataclass(frozen=True)
class Context:
    cache: Cache
    client: openai.AsyncClient
    max_attempts: int
    system_message: openai.types.chat.ChatCompletionSystemMessageParam
    model: str
    statistics: Statistics
    batcher: Batcher
    limit: int
    llm_logger: Logger

    @classmethod
    def create(
        cls,
        *,
        statistics: Statistics,
        model: str,
        language: Language,
        max_tokens: int,
        api_key: str,
        cache_dir: Path | None,
        llm_log_dir: Path | None,
        max_attempts: int,
        limit: int,
    ):
        batcher = sentences_batcher(model, max_tokens)
        cache = Cache.create(cache_dir=cache_dir, language=language)

        system_message = openai.types.chat.ChatCompletionSystemMessageParam(
            role="system",
            content=_get_system_prompt(language),
        )

        client = _create_openai_client(api_key=api_key)
        llm_handler = (
            FileHandler(filename=llm_log_dir / "llm.log")
            if llm_log_dir
            else NullHandler()
        )

        llm_logger = getLogger("llm")
        llm_logger.addHandler(llm_handler)
        llm_logger.setLevel("DEBUG")

        return Context(
            cache=cache,
            client=client,
            max_attempts=max_attempts,
            system_message=system_message,
            model=model,
            statistics=statistics,
            batcher=batcher,
            limit=limit,
            llm_logger=llm_logger,
        )


def translator(
    *,
    model: str,
    language: Language,
    max_tokens: int,
    api_key: str,
    statistics: Statistics,
    cache_dir: Optional[Path] = None,
    llm_log_dir: Optional[Path] = None,
    max_attempts: int = 3,
    limit: int = 0,
) -> Callable[[Iterable[Sentence]], AsyncGenerator[List[TranslatedSubtitle], None]]:
    context = Context.create(
        model=model,
        language=language,
        max_tokens=max_tokens,
        api_key=api_key,
        statistics=statistics,
        cache_dir=cache_dir,
        llm_log_dir=llm_log_dir,
        max_attempts=max_attempts,
        limit=limit,
    )

    async def translate(sentences: Iterable[Sentence]) -> AsyncGenerator[List[TranslatedSubtitle], None]:
        from .batch import translate_batch

        batches = context.batcher(sentences)
        if context.limit > 0:
            batches = islice(batches, context.limit)

        async def mapper(batch: List[Sentence]) -> List[List[TranslatedSubtitle]]:
            return await translate_batch(
                context=context,
                batch=batch,
            )

        async def fallback_mapper(
            sentence: Sentence, exception: TranslatorError
        ) -> List[TranslatedSubtitle]:
            return [
                TranslatedSubtitle.create(
                    start=sub.start,
                    end=sub.end,
                    text="\n".join(
                        [
                            line.strip()
                            for multiline in sub.text
                            for line in multiline.lines
                        ]
                    ),
                )
                for sub in sentence.blocks
            ]

        for batch in batches:
            for result in await adaptive_map(
                batch,
                mapper,
                fallback_mapper,
                TranslatorError,
            ):
                yield result

    return translate
