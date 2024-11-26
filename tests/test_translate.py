import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from srtglot.model import Multiline, Sentence, Subtitle, TranslatedSubtitle
from srtglot.translator import Context, translator
from srtglot.prompt import UserPrompt
from srtglot.languages import Language
from srtglot.config import Config
from bs4 import BeautifulSoup
import pytest


def format_translated(sentences: list[list[TranslatedSubtitle]]) -> str:
    return "\n".join(
        [
            line.text.strip()
            for sentence in sentences
            for line in sentence
            if line.text.strip()
        ]
    )


def test__to_prompt_input():
    batch = [
        Sentence(
            blocks=[
                Subtitle(
                    start=None,
                    end=None,
                    soup=None,
                    text=[
                        Multiline(
                            lines=["Hello", "world", "\n", "How", "are", "you", "?"]
                        )
                    ],
                )
            ]
        )
    ]

    assert (
        UserPrompt.create_prompt(batch).user_message.get("content")
        == "[sentence 1]\nHello\nworld\nHow\nare\nyou\n?"
    )


@pytest.fixture
def sentence() -> Sentence:
    return Sentence(
        blocks=[
            Subtitle(
                start=datetime.time(),
                end=datetime.time(),
                soup=BeautifulSoup(
                    "<i>Hello</i><i>world</i>\n<i>How</i><i>are</i><i>you?</i>",
                    "html.parser",
                ),
                text=[Multiline(lines=["Hello", "world", "", "How", "are", "you?"])],
            )
        ]
    )


def create_context() -> Context:
    return Context.create(
        config=Config(
            model="gpt-4o",
            target_language=Language.EN,
            max_tokens=100,
            api_key="sk-xxx",
            max_attempts=3,
            input=Path("input.srt"),
            output=Path("output.srt"),
        )
    )


@pytest.mark.asyncio
async def test_should_get_llm_completions_when_cache_is_missing(sentence: Sentence):
    with patch("srtglot.context._create_openai_client") as create_client:
        client = AsyncMock(name="client")
        create_client.return_value = client

        choice = AsyncMock(name="choice")
        choice.message.content = "[sentence 1]\nBonjour\nmonde\nComment\nça\nva?"

        completion = AsyncMock(name="completion")
        completion.choices = [choice]
        client.chat.completions.create.return_value = completion

        translate = translator(create_context())

        result = await translate([sentence])
        assert (
            format_translated(result)
            == "<i>Bonjour</i><i>monde</i>\n<i>Comment</i><i>ça</i><i>va?</i>"
        )


@pytest.mark.asyncio
async def test_should_get_llm_completions_from_cache_when_cache_is_present(
    sentence: Sentence,
):
    with patch("srtglot.context.Cache.create") as cache:
        cache.return_value.get.return_value = [
            [
                TranslatedSubtitle(
                    start="00:00:00,000",
                    end="00:00:00,000",
                    text="Bonjour\nmonde\nComment\nça\nva?",
                )
            ]
        ]

        translate = translator(create_context())
        result = await translate([sentence])
        assert format_translated(result) == "Bonjour\nmonde\nComment\nça\nva?"
