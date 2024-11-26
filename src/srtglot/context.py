from dataclasses import dataclass
from collections.abc import Callable

import openai

from .model import Sentence
from .sentence import sentences_batcher, Batcher
from .cache import Cache
from .config import Config
from .prompt import get_system_prompt, UserPrompt
from .logging import setup_llm_logging


class TranslatorError(ValueError):
    batch: list[Sentence]
    completions: list[str]

    def __init__(self, batch: list[Sentence], completions: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)


def _create_openai_client(*, api_key) -> openai.AsyncClient:
    return openai.AsyncClient(api_key=api_key)


@dataclass(frozen=True)
class Context:
    config: Config
    cache: Cache
    batcher: Batcher
    client: openai.AsyncClient
    system_message: openai.types.chat.ChatCompletionSystemMessageParam
    llm_logger: Callable[[UserPrompt, str | None], None]

    @classmethod
    def create(
        cls,
        *,
        config: Config,
    ):
        return cls(
            config=config,
            client=_create_openai_client(api_key=config.api_key),
            system_message=get_system_prompt(config),
            batcher=sentences_batcher(config.model, config.max_tokens),
            llm_logger=setup_llm_logging(config),
            cache=Cache.create(
                cache_dir=config.cache_dir, language=config.target_language
            ),
        )
