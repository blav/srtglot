from unittest.mock import MagicMock, patch
from srt_gpt_translator.model import Multiline, Sentence, Subtitle, TranslatedSubtitle
from srt_gpt_translator.translator import _to_prompt_input, translator


def format_translated(translated_text: list[TranslatedSubtitle]) -> str:
    return " ".join(
        [
            line
            for sub in translated_text
            for line in sub.translated_text
            if line.strip()
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

    assert _to_prompt_input(batch) == "Hello\nworld\nHow\nare\nyou\n?"


def test_translator():
    with patch("srt_gpt_translator.translator._create_openai_client") as create_client:
        client = MagicMock(name="client")
        create_client.return_value = client

        choice = MagicMock(name="choice")
        choice.text = "Bonjour\nmonde\nComment\nça\nva?"

        completion = MagicMock(name="completion")
        completion.choices = [choice]
        client.completions.create.return_value = completion

        sentence = Sentence(
            blocks=[
                Subtitle(
                    start=None,
                    end=None,
                    soup=None,
                    text=[
                        Multiline(
                            lines=["Hello", "world", "", "How", "are", "you?"]
                        )
                    ],
                )
            ]
        )

        translate = translator(
            model="gpt-4o",
            language="french",
            max_tokens=100,
            api_key="sk-xxx",
        )

        result = [*translate([sentence])]
        assert format_translated(result) == "Bonjour\nmonde\n\nComment\nça\nva?"
