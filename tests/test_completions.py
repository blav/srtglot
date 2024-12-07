from datetime import time

from unittest.mock import MagicMock
from srtglot.completions import parse_completions, map_to_translated_subtitle
from srtglot.context import TranslatorError
from srtglot.model import Multiline, Sentence, Subtitle, TranslatedSubtitle

from bs4 import BeautifulSoup


def test_should_parse_completions():
    content = """[sentence 1]
    Hello, world!
    Bonjour, tout le monde!

    [sentence 2]
    Goodbye, world!
    Au revoir, tout le monde!
    """

    sentence1 = MagicMock(spec=Sentence)
    sentence1.non_empty_text_lines_count = 2
    sentence2 = MagicMock(spec=Sentence)
    sentence2.non_empty_text_lines_count = 2
    completions = parse_completions([sentence1, sentence2], content)
    assert completions == [
        ["Hello, world!", "Bonjour, tout le monde!"],
        ["Goodbye, world!", "Au revoir, tout le monde!"],
    ]


def test_should_raise_when_delimiter_is_missing():
    content = """
    Hello, world!
    Bonjour, tout le monde!
    """
    try:
        parse_completions([], content)
        assert False
    except TranslatorError as e:
        assert str(e) == "Delimiter expected, got Hello, world!"
        assert e.batch == []
        assert e.completions == ["Hello, world!", "Bonjour, tout le monde!"]


def test_should_raise_when_number_of_sentences_does_not_match_completions():
    content = """
    [sentence 1]
    Hello, world!
    """
    try:
        parse_completions([], content)
        assert False
    except TranslatorError as e:
        assert (
            str(e)
            == "Number of sentences in prompt and completions must match. Prompt: 0, Completions: 1"
        )


def test_should_raise_when_number_of_sentence_fragments_differ_by_2_from_completions():
    content = """
    [sentence 1]
    Hello, world!
    Bonjour, tout le monde!
    coucou
    """

    sentence = Sentence(blocks=[])
    try:
        parse_completions([sentence], content)
        assert False
    except TranslatorError as e:
        assert (
            str(e)
            == "Number of sentence fragments differ by more than 2 from completions. Prompt: 0, Completions: 3"
        )


def test_should_raise_when_completion_is_empty():
    content = """
    [sentence 1]
    """

    sentence = Sentence(blocks=[])
    try:
        parse_completions([sentence], content)
        assert False
    except TranslatorError as e:
        assert str(e) == "Empty completion"
        assert e.batch == [sentence]
        assert e.completions == []


def test_should_map_to_translated_subtitles():
    now = time(0, 0, 12, 178000)
    sentence1 = Sentence(
        blocks=[
            Subtitle(
                start=now,
                end=now,
                soup=BeautifulSoup("<i></i><i>S1</i><i></i>", "html.parser"),
                text=[
                    Multiline(
                        lines=[
                            "",
                            "A1",
                            "",
                        ]
                    ),
                ],
            ),
            Subtitle(
                start=now,
                end=now,
                soup=BeautifulSoup("<i></i><i>S2</i><i>s3</i><i></i>", "html.parser"),
                text=[
                    Multiline(
                        lines=[
                            "",
                            "A2",
                            "A3",
                            "",
                        ]
                    ),
                ],
            ),
        ]
    )

    assert map_to_translated_subtitle(
        sentences=[sentence1],
        parsed_completions=[["B1", "B2", "B3"]],
        attempt_number=1,
    ) == [
        TranslatedSubtitle(
            start="00:00:12,178",
            end="00:00:12,178",
            text="<i></i><i>A1</i><i></i>",
        ),
        TranslatedSubtitle(
            start="00:00:12,178",
            end="00:00:12,178",
            text="<i></i><i>S2</i><i>s3</i><i></i>",
        ),
    ]
