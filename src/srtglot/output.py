from io import TextIOBase
from typing import Iterable, List, TextIO

from .model import TranslatedSubtitle


def output_text(input: Iterable[List[TranslatedSubtitle]], output: TextIO):
    for sentence in input:
        text = [subtitle.text.replace("\n", " ").strip() for subtitle in sentence]
        output.write(" ".join([t for t in text if t]))
        output.write("\n")


def output_srt(input: Iterable[TranslatedSubtitle], output: TextIO):
    for i, subtitle in enumerate(input):
        output.write(str(i + 1))
        output.write("\n")
        output.write(f"{subtitle.start} --> {subtitle.end}")
        output.write("\n")
        output.write(subtitle.text)
        output.write("\n\n")
