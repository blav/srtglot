import os
from pathlib import Path
from dataclasses import dataclass

import click

from srtglot.languages import Language


@dataclass
class Config:
    input: Path
    output: Path
    model: str
    target_language: Language
    api_key: str
    parallelism: int = 20
    max_tokens: int = 100
    cache_dir: Path | None = None
    llm_log_dir: Path | None = None
    max_attempts: int = 3
    limit: int = 0

    @classmethod
    def create_config(
        cls,
        *,
        input: Path,
        output: Path,
        target_language: str,
        model: str = "gpt-4o",
        parallelism: int = 20,
        max_tokens: int = 100,
        cache_dir: Path | None = None,
        llm_log_dir: Path | None = None,
        max_attempts: int = 3,
        limit: int = 0,
    ) -> "Config":
        api_key = os.environ["OPENAI_API_KEY"]
        if not api_key:
            raise click.ClickException(
                "Please set the OPENAI_API_KEY environment variable or .env file with your OpenAI API key."
            )

        if not input:
            raise click.ClickException("Please provide a valid input file path.")

        if not output:
            raise click.ClickException("Please provide a valid output file path.")

        if not target_language:
            raise click.ClickException("Please provide a valid target language.")

        return Config(
            input=input,
            output=output,
            model=model,
            api_key=api_key,
            parallelism=parallelism,
            target_language=Language[target_language.upper()],
            max_tokens=max_tokens,
            cache_dir=cache_dir.expanduser().resolve() if cache_dir else None,
            llm_log_dir=llm_log_dir.expanduser().resolve() if llm_log_dir else None,
            max_attempts=max_attempts,
            limit=limit,
        )
