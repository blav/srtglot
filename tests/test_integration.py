import os
from pathlib import Path
from srt_gpt_translator.translator import translator
from srt_gpt_translator.parser import parse
from srt_gpt_translator.sentence import collect_sentences
from fixtures import srt_file


def _test_integration(srt_file: Path):
    model = os.environ.get("OPENAI_MODEL")
    translate = translator(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        language="french",
        max_tokens=1000,
    )

    subtitles = parse(srt_file)
    sentences=[*collect_sentences(subtitles=subtitles)][:50]
    translated=[*translate(sentences)]
    print(translated)