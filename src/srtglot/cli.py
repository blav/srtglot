from functools import reduce
from itertools import islice
from operator import add
import os
import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator, Generator
import aiofiles
import rich_click as click

from .parser import parse
from .translator import Context, translator
from .sentence import collect_sentences
from .languages import Language
from .statistics import Statistics
from .renderer import render_srt
from .model import Sentence, TranslatedSubtitle
from rich.progress import Progress


@click.command()
@click.option(
    "--target-language",
    "-t",
    required=True,
    help="The target language to translate the subtitle text into.",
    type=click.Choice([lang.name.lower() for lang in Language]),
)
@click.option(
    "--input",
    "-i",
    help="The srt input file to translate.",
    required=True,
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=Path
    ),
)
@click.option(
    "--output",
    "-o",
    help="The translated srt output file.",
    required=True,
    type=click.Path(
        exists=False, dir_okay=False, file_okay=True, writable=True, path_type=Path
    ),
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=os.environ.get("LIMIT", 0),
    show_default=True,
    help="Only translate the first N sentences.",
)
@click.option(
    "--model",
    "-m",
    help="The model to use for translation.",
    default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    show_default=True,
)
@click.option(
    "--max-tokens",
    "-x",
    help="Sentence batch max size in tokens.",
    default=os.environ.get("MAX_TOKENS", 100),
    show_default=True,
    type=int,
)
@click.option(
    "--cache-dir",
    "-c",
    help="Cache directory for storing language completions.",
    default=os.environ.get("CACHE_DIR", "~/.cache/srtglot"),
    show_default=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--max-attempts",
    "-a",
    help="Max number of attempts when translating a sentence batch.",
    default=os.environ.get("MAX_ATTEMPTS", 3),
    show_default=True,
    type=int,
)
@click.option(
    "--llm-log-dir",
    "-d",
    help="Log directory for storing llm logs.",
    default=os.environ.get("LLM_LOG_DIR"),
    show_default=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "--parallelism",
    "-p",
    help="Number of sentence bacthes to translate in parallel.",
    default=os.environ.get("PARALLELISM", 20),
    show_default=True,
    type=int,
)
def main(
    target_language: str,
    input: Path,
    output: Path,
    limit: int,
    model: str,
    max_tokens: int,
    cache_dir: Path,
    llm_log_dir: Path,
    max_attempts: int,
    parallelism: int,
):
    api_key = os.environ["OPENAI_API_KEY"]
    if not api_key:
        raise click.ClickException(
            "Please set the OPENAI_API_KEY environment variable or .env file with your OpenAI API key."
        )

    language = Language[target_language.upper()]

    statistics = Statistics()
    context = Context.create(
        model=model,
        language=language,
        max_tokens=max_tokens,
        api_key=api_key,
        statistics=statistics,
        cache_dir=cache_dir.expanduser().resolve() if cache_dir else None,
        llm_log_dir=llm_log_dir.expanduser().resolve() if llm_log_dir else None,
        max_attempts=max_attempts,
        limit=limit,
    )

    translate = translator(context)

    subtitles = [*parse(input)]
    sentences = collect_sentences(iter(subtitles))
    batches = context.batcher(sentences)
    if context.limit > 0:
        batches = islice(batches, context.limit)

    async def mainloop():
        async with aiofiles.open(output, "w") as output_stream:

            def batched() -> Generator[list[list[Sentence]], None, None]:
                while batch := list(islice(batches, parallelism)):
                    yield batch

            async def translate_batch(
                batch: list[Sentence],
            ) -> list[list[TranslatedSubtitle]]:
                result = await translate(batch)
                advance = reduce(add, [len(b) for b in result])
                progress.update(task, advance=advance)
                return result

            async def subtitles_iter() -> AsyncGenerator[TranslatedSubtitle, None]:
                for batch in batched():
                    tasks = [asyncio.create_task(translate_batch(b)) for b in batch]
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        for subtitles_list in result:
                            for sentence in subtitles_list:
                                yield sentence

            await render_srt(input=subtitles_iter(), output=output_stream)

    try:
        with Progress() as progress:
            task = progress.add_task("Translating subtitles...", total=len(subtitles))
            asyncio.run(mainloop())
    finally:
        print(statistics)


if __name__ == "__main__":
    main()
