from pathlib import Path
from typing import Generator, Iterable
from dataclasses import dataclass
from pysrt import open as open_srt
from pysrt.srtitem import SubRipItem
from bs4 import BeautifulSoup
from .model import Subtitle, Multiline


def parse(input: Path) -> Generator[Subtitle, None, None]:
    subs: Iterable[SubRipItem] = open_srt(input)
    for sub in subs:
        soup = BeautifulSoup(sub.text, "html.parser")
        yield Subtitle(
            start=sub.start.to_time(),
            end=sub.end.to_time(),
            soup=soup,
            text=[
                Multiline(block.strip().split("\n"))
                for block in soup.find_all(string=True)
            ],
        )
