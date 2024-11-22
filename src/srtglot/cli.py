import os
import sys
from pathlib import Path
from typing import Iterable
import rich_click as click
from .parser import parse
from .translator import translator
from .sentence import collect_sentences
from .languages import Language
from .statistics import Statistics
from .renderer import render_srt
from .model import TranslatedSubtitle
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
    required=False,
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
):
    api_key = os.environ["OPENAI_API_KEY"]
    if not api_key:
        raise click.ClickException(
            "Please set the OPENAI_API_KEY environment variable or .env file with your OpenAI API key."
        )

    language = Language[target_language.upper()]

    statistics = Statistics()
    translate = translator(
        model=model,
        language=language,
        max_tokens=max_tokens,
        limit=limit,
        api_key=api_key,
        cache_dir=cache_dir.expanduser().resolve() if cache_dir else None,
        llm_log_dir=llm_log_dir.expanduser().resolve() if llm_log_dir else None,
        max_attempts=max_attempts,
        statistics=statistics,
    )

    subtitles = [*parse(input)]
    sentences = collect_sentences(iter(subtitles))

    try:
        with Progress() as progress:
            task = progress.add_task("Translating subtitles...", total=len(subtitles))

            def subtitles_iter() -> Iterable[TranslatedSubtitle]:
                for batch in translate(sentences):
                    yield from batch

            def update_progress(subtitle: TranslatedSubtitle) -> TranslatedSubtitle:
                progress.update(task, advance=1)
                return subtitle

            translated_sub = map(update_progress, subtitles_iter())
            if output:
                with output.open("w") as f:
                    render_srt(input=translated_sub, output=f)
            else:
                render_srt(input=translated_sub, output=sys.stdout)
                
    finally:
        print(statistics)


if __name__ == "__main__":
    main()
