from pathlib import Path
from unittest.mock import MagicMock
from srtglot.config import Config
from srtglot.prompt import get_system_prompt
from srtglot.languages import Language


def test_get_system_prompt():
    config = MagicMock(spec=Config)
    config.target_language = Language.FR

    system_prompt = get_system_prompt(config)
    assert system_prompt["role"] == "system"
    assert (
        "You will translate input sentences from any language to the French language."
        in system_prompt["content"]
    )
