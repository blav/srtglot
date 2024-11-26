from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from logging import getLogger, FileHandler, NullHandler, Logger

import openai
from jinja2 import Template

from .model import Sentence
from .sentence import sentences_batcher, Batcher
from .cache import Cache
from .languages import Language
from .statistics import Statistics
from .config import Config


class TranslatorError(ValueError):
    batch: list[Sentence]
    completions: list[str]

    def __init__(self, batch: list[Sentence], completions: list[str], *args, **kwargs):
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
    config: Config
    cache: Cache
    batcher: Batcher
    statistics: Statistics
    client: openai.AsyncClient
    llm_logger: Logger
    system_message: openai.types.chat.ChatCompletionSystemMessageParam

    @classmethod
    def create(
        cls,
        *,
        config: Config,
    ):
        batcher = sentences_batcher(config.model, config.max_tokens)
        cache = Cache.create(
            cache_dir=config.cache_dir, language=config.target_language
        )

        system_message = openai.types.chat.ChatCompletionSystemMessageParam(
            role="system",
            content=_get_system_prompt(config.target_language),
        )

        client = _create_openai_client(api_key=config.api_key)
        llm_handler = (
            FileHandler(filename=config.llm_log_dir / "llm.log")
            if config.llm_log_dir
            else NullHandler()
        )

        llm_logger = getLogger("llm")
        llm_logger.addHandler(llm_handler)
        llm_logger.setLevel("DEBUG")

        return Context(
            config=config,
            cache=cache,
            client=client,
            system_message=system_message,
            statistics=Statistics(),
            batcher=batcher,
            llm_logger=llm_logger,
        )
