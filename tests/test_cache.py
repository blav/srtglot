from datetime import time
from pathlib import Path
import json
from srtglot.cache import Cache
from srtglot.model import Sentence, TranslatedSubtitle, Subtitle, Multiline
from srtglot.languages import Language
from tempfile import TemporaryDirectory
from bs4 import BeautifulSoup
import pytest


@pytest.fixture
def sentences() -> list[Sentence]:
    return [
        Sentence(
            blocks=[
                Subtitle(
                    start=time(),
                    end=time(),
                    soup=BeautifulSoup("Hello\nworld!", features="html.parser"),
                    text=[Multiline(lines=["Hello", "world!"])],
                ),
            ]
        ),
        Sentence(
            blocks=[
                Subtitle(
                    start=time(),
                    end=time(),
                    soup=BeautifulSoup("Hello\nworld 2!", features="html.parser"),
                    text=[Multiline(lines=["Hello", "world 2!"])],
                ),
            ]
        ),
    ]


def test_should_get_none_if_cache_dir_is_none():
    cache = Cache.create(cache_dir=None, language=Language.FR)
    assert cache.get([Sentence]) == None


def test_test_should_put_and_get_json_file(sentences: list[Sentence]):
    now = time()
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache = Cache.create(cache_dir=cache_dir, language=Language.FR)
        key = sentences

        value = [
            [
                TranslatedSubtitle(
                    start="00:00:00,000",
                    end="00:00:00,000",
                    text="Hello\nworld!",
                )
            ],
            [
                TranslatedSubtitle(
                    start="00:00:00,000",
                    end="00:00:00,000",
                    text="Hello\nworld 2!",
                )
            ],
        ]

        cache.put(key, value)
        assert cache.get(key) == value

        value_file_1 = (
            cache_dir / "FR" / "8006426e8c4301d4859b01a26304897c47261d64.json"
        )
        assert value_file_1.is_file()

        assert json.loads(value_file_1.read_text()) == [
            {
                "end": "00:00:00,000",
                "start": "00:00:00,000",
                "text": "Hello\nworld 2!",
            }
        ]

        value_file_2 = (
            cache_dir / "FR" / "a909886ba52ff4a0014e894499a1ce8a9996983c.json"
        )
        assert value_file_2.is_file()

        assert json.loads(value_file_2.read_text()) == [
            {
                "end": "00:00:00,000",
                "start": "00:00:00,000",
                "text": "Hello\nworld!",
            }
        ]

