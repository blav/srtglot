from typing import Iterable, TextIO

from .model import TranslatedSubtitle


def render_srt(input: Iterable[TranslatedSubtitle], output: TextIO):
    for i, subtitle in enumerate(input):
        output.write(str(i + 1))
        output.write("\n")
        output.write(f"{subtitle.start} --> {subtitle.end}")
        output.write("\n")
        output.write(subtitle.text)
        output.write("\n\n")
