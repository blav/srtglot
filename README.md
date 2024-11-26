
# SRTGlot

SRTGlot is a Python-based tool for translating subtitle files (.srt) into various languages. The project leverages modern language models and a modular architecture to provide seamless translation workflows for subtitles.

## Features
- **Multi-language Support**: Translate subtitles to various languages using OpenAI models.
- **Batch Processing**: Handles large subtitle files with customizable batching.
- **Customizable**: Fine-tune translation parameters such as model, token limits, and parallelism.
- **CLI Integration**: Simple and flexible command-line interface for all functionalities.

## Installation

### Using Docker
This project includes a Docker setup for easy deployment:
1. Build the Docker image:
   ```bash
   docker build -t srtglot .
   ```
2. Run the container:
   ```bash
   docker run --rm -v $(pwd):/app srtglot --help
   ```

### Using Poetry
1. Install [Poetry](https://python-poetry.org/).
2. Clone the repository and navigate to its directory:
   ```bash
   git clone git@github.com:blav/srtglot.git
   cd srtglot
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```

## Usage

### CLI Options
Run the following command to see all available options:
```bash
srtglot --help
```

#### Example Command
Translate a subtitle file into French:
```bash
srtglot -i input.srt -o output.srt -t fr
```

### Parameters
- `--target-language (-t)`: Target language for translation (e.g., `fr`, `es`).
- `--input (-i)`: Path to the input `.srt` file.
- `--output (-o)`: Path to save the translated `.srt` file.
- Additional options like `--limit`, `--model`, `--max-tokens`, etc., allow fine-grained control over translations.

## Development

### Project Structure
- `pyproject.toml`: Defines project metadata and dependencies.
- `cli.py`: Entry point for the CLI, utilizing `click` for argument parsing.
- `prompt.jinja`: Template for constructing prompts for the language model.

### Running Tests
This project uses `pytest` for testing:
```bash
poetry run pytest
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- Built with OpenAI API, Click, and Jinja2.
- Inspired by the need for seamless multilingual subtitle translation.

## Contribution
Contributions are welcome! Fork the repository and submit a pull request for review.

---

**Author**: blav
