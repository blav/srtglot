import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from .model import Sentence, TranslatedSubtitle
from .languages import Language


@dataclass(frozen=True)
class Cache:
    cache_dir: Path | None

    def __post_init__(self):
        if self.cache_dir is None:
            return
        
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        if not self.cache_dir.is_dir():
            raise ValueError(f"{self.cache_dir} is not a directory")

    def get(self, key: list[Sentence]) -> list[TranslatedSubtitle] | None:
        if self.cache_dir is None:
            return None
        
        entry_path = self._to_entry_path(key)
        if not entry_path.exists():
            return None

        with entry_path.open("r") as f:
            return [TranslatedSubtitle(**item) for item in json.load(f)]

    def put(self, key: list[Sentence], value: list[TranslatedSubtitle]):
        if self.cache_dir is None:
            return
        
        entry_path = self._to_entry_path(key)
        with entry_path.open("w") as f:
            json.dump([vars(subtitle) for subtitle in value], f)

    def _to_entry_path(self, key: list[Sentence]) -> Path:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")
        
        sha1 = hashlib.sha1()
        for sentence in key:
            for block in sentence.blocks:
                for multiline in block.text:
                    for line in multiline.lines:
                        sha1.update(line.encode())

        return self.cache_dir / (sha1.hexdigest() + ".json")
    
    @staticmethod
    def create(cache_dir: Path | None, language: Language) -> "Cache":
        if cache_dir is None:
            return Cache(cache_dir=None)
        
        cache_dir = cache_dir / language.name
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        return Cache(cache_dir=cache_dir)
