import datetime
from unittest.mock import MagicMock, patch
from srt_gpt_translator.model import Multiline, Sentence, Subtitle, TranslatedSubtitle
from srt_gpt_translator.translator import _to_prompt_input, translator
from srt_gpt_translator.languages import Language
from srt_gpt_translator.statistics import Statistics
from bs4 import BeautifulSoup
import pytest


def format_translated(subs: list[TranslatedSubtitle]) -> str:
    return "\n".join([line.text.strip() for line in subs if line.text.strip()])


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

    assert _to_prompt_input(batch) == "[sentence 1]\nHello\nworld\nHow\nare\nyou\n?"


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


@pytest.fixture
def translator_params() -> dict:
    return {
        "model": "gpt-4o",
        "language": Language.EN,
        "max_tokens": 100,
        "api_key": "sk-xxx",
        "max_attempts": 3,
        "statistics": Statistics(),
    }


def test_should_get_llm_completions_when_cache_is_missing(
    sentence: Sentence, translator_params: dict
):
    with patch("srt_gpt_translator.translator._create_openai_client") as create_client:
        client = MagicMock(name="client")
        create_client.return_value = client

        choice = MagicMock(name="choice")
        choice.message.content = "[sentence 1]\nBonjour\nmonde\nComment\nça\nva?"

        completion = MagicMock(name="completion")
        completion.choices = [choice]
        client.chat.completions.create.return_value = completion

        translate = translator(**translator_params)

        result = [*translate([sentence])]
        assert (
            format_translated(result)
            == "<i>Bonjour</i><i>monde</i>\n<i>Comment</i><i>ça</i><i>va?</i>"
        )


def test_should_get_llm_completions_from_cache_when_cache_is_present(
    sentence: Sentence, translator_params: dict
):
    with patch("srt_gpt_translator.translator.Cache.create") as cache:
        cache.return_value.get.return_value = [
            TranslatedSubtitle(
                start="00:00:00,000",
                end="00:00:00,000",
                text="Bonjour\nmonde\nComment\nça\nva?",
            )
        ]

        translate = translator(**translator_params)
        result = [*translate([sentence])]
        assert format_translated(result) == "Bonjour\nmonde\nComment\nça\nva?"
