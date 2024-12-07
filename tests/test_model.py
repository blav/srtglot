from datetime import time
from pathlib import Path
from srtglot.model import Sentence, Subtitle, Multiline
from srtglot.parser import parse
from fixtures import srt_file


def test_should_translate_subtitle(srt_file: Path):
    sub = [*parse(srt_file)][0]
    assert sub.text_lines == ["As the first century", "", "of the Targaryen dynasty"]
    translated = sub.translate(
        ["AS THE FIRST CENTURY", "\n", "OF THE TARGARYEN DYNASTY"]
    )

    assert (
        translated.text
        == "<i>AS THE FIRST CENTURY</i>\n<i>OF THE TARGARYEN DYNASTY</i>"
    )


def test_should_return_lines():
    sub = Subtitle(
        start=time(0, 0, 12, 178000),
        end=time(0, 0, 14, 848000),
        text=[
            Multiline(["As the ", "first century"]),
            Multiline(["of the Targaryen dynasty"]),
        ],
        soup=None,
    )

    assert sub.text_lines == ["As the ", "first century", "of the Targaryen dynasty"]


def test_should_return_sentence_lines():
    assert Sentence(
        blocks=[
            Subtitle(
                start=None,
                end=None,
                soup=None,
                text=[
                    Multiline(
                        lines=[
                            "As the ",
                        ]
                    ),
                    Multiline(
                        lines=[
                            "first century",
                            "of the Targaryen dynasty",
                        ]
                    ),
                ],
            ),
            Subtitle(
                start=None,
                end=None,
                soup=None,
                text=[
                    Multiline(
                        lines=[
                            "the last dragon",
                            "has been killed.",
                        ]
                    ),
                ],
            ),
        ]
    ).text_lines == [
        "As the ",
        "first century",
        "of the Targaryen dynasty",
        "the last dragon",
        "has been killed.",
    ]


def test_should_return_sentence_length():
    sentence = Sentence(
        blocks=[
            Subtitle(
                start=None,
                end=None,
                soup=None,
                text=[
                    Multiline(
                        lines=[
                            "As the ",
                            "",
                        ]
                    ),
                    Multiline(
                        lines=[
                            "first century",
                            "of the Targaryen dynasty",
                        ]
                    ),
                ],
            ),
            Subtitle(
                start=None,
                end=None,
                soup=None,
                text=[
                    Multiline(
                        lines=[
                            "the last dragon",
                            "has been killed.",
                        ]
                    ),
                ],
            ),
        ]
    )
    assert sentence.non_empty_text_lines_count == 5
