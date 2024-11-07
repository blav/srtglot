from pathlib import Path
import rich_click as click
import pysrt
from .parser import parse


@click.command()
@click.option(
    "--target-language",
    "-l",
    required=True,
    help="The target language to translate the subtitle text into (eg. Chinese, French).",
)
@click.option(
    "--input",
    "-i",
    help="The input srt file to translate.",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option(
    "--test", "-t", is_flag=True, help="Only translate the first 3 short texts."
)
@click.option(
    "--model", "-m", help="The model to use for translation.", default="gpt-4o-mini"
)
def main(target_language: str, input: Path, test: bool, model: str):
    parsed_subs = list(parse(input))
    if test:
        parsed_subs = parsed_subs[:3]

    for sub in parsed_subs:
        print(sub.text)


if __name__ == "__main__":
    main()
