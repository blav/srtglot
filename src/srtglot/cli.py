import os
from pathlib import Path
import rich_click as click
from .parser import parse
from .translator import translator
from .sentence import collect_sentences
from .languages import Language
from .statistics import Statistics
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
    help="The input srt file to translate.",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
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
    default=os.environ.get("CACHE_DIR", "~/.cache/srtx"),
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
def main(
    target_language: str,
    input: Path,
    limit: int,
    model: str,
    max_tokens: int,
    cache_dir: Path,
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
        cache_dir=Path(os.environ.get("CACHE_DIR", "~/.cache/srtx")),
        max_attempts=max_attempts,
        statistics=statistics,
    )

    subtitles = [*parse(input)]
    sentences = collect_sentences(iter(subtitles))

    try:
        with Progress() as progress:
            task = progress.add_task("Translating subtitles...", total=len(subtitles))
            for translated_sub in translate(sentences):
                progress.update(task, advance=1)
                print(translated_sub)
    finally:
        print(statistics)


if __name__ == "__main__":
    main()
