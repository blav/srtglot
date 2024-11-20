from pathlib import Path
from srtglot import parser
from srtglot.model import Subtitle, Multiline
from datetime import time
from fixtures import srt_file


def test_parse(srt_file: Path):
    subtitles = [*parser.parse(srt_file)]
    assert len(subtitles) == 690
    assert subtitles[0] == Subtitle(
        start=time(0, 0, 12, 178000),
        end=time(0, 0, 14, 848000),
        text=[
            Multiline(lines=["As the first century"]),
            Multiline(lines=[""]),
            Multiline(lines=["of the Targaryen dynasty"]),
        ],
        soup=subtitles[0].soup,
    )
