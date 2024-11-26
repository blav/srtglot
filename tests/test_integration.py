import os
from pathlib import Path

from srtglot.languages import Language
from srtglot.translator import translator, Context
from srtglot.parser import parse
from srtglot.sentence import collect_sentences
from srtglot.config import Config

import pytest


@pytest.mark.asyncio
async def _test_integration(srt_file: Path):
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("OPENAI_MODEL environment variable is not set.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    translate = translator(
        Context.create(
            config=Config(
                model=model,
                api_key=api_key,
                target_language=Language.FR,
                max_tokens=1000,
                input=srt_file,
                output=Path("output.srt"),
            )
        )
    )

    subtitles = parse(srt_file)
    sentences = [*collect_sentences(subtitles=subtitles)][:50]
    translated = await translate(sentences)
    print(translated)
