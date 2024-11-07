from pathlib import Path
import pytest

@pytest.fixture
def srt_file() -> Path:
    return Path(__file__).parent / "hod.srt"
