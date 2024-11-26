# SRTGlot

SRTGlot is a command-line tool for translating subtitle files (.srt) using OpenAI's language models.

## Features

- Translate subtitle files to different languages
- Supports multiple translation models
- Batch processing of sentences
- Progress tracking with rich progress bars

## Installation

### Using Docker

You can build and run the Docker image:

```sh
docker build -t srtglot .
docker run --rm -v $(pwd):/app srtglot --help
