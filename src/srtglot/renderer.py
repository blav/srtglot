from collections.abc import AsyncGenerator
from aiofiles.threadpool.text import AsyncTextIOWrapper
from .model import TranslatedSubtitle


async def render_srt(
    input: AsyncGenerator[TranslatedSubtitle, None], output: AsyncTextIOWrapper
):
    index = 0
    async for subtitle in input:
        await output.write(str(index := index + 1))
        await output.write("\n")
        await output.write(f"{subtitle.start} --> {subtitle.end}")
        await output.write("\n")
        await output.write(subtitle.text)
        await output.write("\n\n")
