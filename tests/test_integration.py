import os
from pathlib import Path
from srtglot.languages import Language
from srtglot.statistics import Statistics
from srtglot.translator import translator
from srtglot.parser import parse
from srtglot.sentence import collect_sentences
from fixtures import srt_file


def _test_integration(srt_file: Path):
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise ValueError("OPENAI_MODEL environment variable is not set.")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    translate = translator(
        model=model,
        api_key=api_key,
        language=Language.FR,
        max_tokens=1000,
        statistics=Statistics(),
    )

    subtitles = parse(srt_file)
    sentences = [*collect_sentences(subtitles=subtitles)][:50]
    translated = [*translate(sentences)]
    print(translated)
