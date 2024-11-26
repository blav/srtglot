import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles

from .model import Sentence, TranslatedSubtitle
from .languages import Language


@dataclass(frozen=True)
class Cache:
    cache_dir: Path | None

    async def get(self, key: list[Sentence]) -> list[list[TranslatedSubtitle]] | None:
        if self.cache_dir is None:
            return None

        cached = []
        for sentence in key:
            entry_path = self._to_entry_path(sentence)
            if not entry_path.exists():
                return None

            async with aiofiles.open(entry_path, "r") as f:
                cached.append(
                    [TranslatedSubtitle(**item) for item in json.loads(await f.read())]
                )

        return cached

    async def put(self, key: list[Sentence], batch: list[list[TranslatedSubtitle]]):
        if self.cache_dir is None:
            return

        for sentence, subtitles in zip(key, batch):
            entry_path = self._to_entry_path(sentence)
            async with aiofiles.open(entry_path, "w") as f:
                await f.write(json.dumps([asdict(subtitle) for subtitle in subtitles]))

    def _to_entry_path(self, sentence: Sentence) -> Path:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")

        sha1 = hashlib.sha1()
        for block in sentence.blocks:
            for multiline in block.text:
                for line in multiline.lines:
                    sha1.update(line.encode())

        return self.cache_dir / (sha1.hexdigest() + ".json")

    @classmethod
    def create(cls, cache_dir: Path | None, language: Language) -> "Cache":
        if cache_dir is not None:
            cache_dir = cache_dir.expanduser().resolve() / language.name
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)

            if not cache_dir.is_dir():
                raise ValueError(f"{cache_dir} is not a directory")

        return cls(cache_dir=cache_dir)
