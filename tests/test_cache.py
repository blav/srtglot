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
def sentence():
    return Sentence(
        blocks=[
            Subtitle(
                start=time(),
                end=time(),
                soup=BeautifulSoup("Hello\nworld!", features="html.parser"),
                text=[Multiline(lines=["Hello", "world!"])],
            )
        ]
    )


def test_should_get_none_if_cache_dir_is_none():
    cache = Cache.create(cache_dir=None, language=Language.FR)
    assert cache.get([Sentence]) == None


def test_test_should_put_and_get_json_file(sentence: Sentence):
    now = time()
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache = Cache.create(cache_dir=cache_dir, language=Language.FR)
        key = [sentence]

        value = [
            [
                TranslatedSubtitle(
                    start="00:00:00,000",
                    end="00:00:00,000",
                    text="Hello\nworld!",
                )
            ]
        ]

        cache.put(key, value)
        assert cache.get(key) == value

        value_file = cache_dir / "fr" / "a909886ba52ff4a0014e894499a1ce8a9996983c.json"
        assert value_file.is_file()

        assert json.loads(value_file.read_text()) == [
            [
                {
                    "end": "00:00:00,000",
                    "start": "00:00:00,000",
                    "text": "Hello\nworld!",
                }
            ]
        ]

        assert cache.get(key) == value
