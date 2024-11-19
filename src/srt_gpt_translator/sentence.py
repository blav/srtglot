from typing import Callable, Iterable, List
import tiktoken
from .model import Subtitle, Sentence


def collect_sentences(subtitles: Iterable[Subtitle]) -> Iterable[Sentence]:
    blocks = []
    for sub in subtitles:
        if sub.text[-1].lines[-1].strip().endswith("."):
            blocks.append(sub)
            yield Sentence(blocks)
            blocks = []
        else:
            blocks.append(sub)

    if blocks:
        yield Sentence(blocks)


def token_counter(model: str) -> Callable[[Sentence], int]:
    encoding = tiktoken.encoding_for_model(model)

    def count_tokens(sentence: Sentence) -> int:
        return len(encoding.encode(str(sentence)))

    return count_tokens


def sentences_batcher(
    model: str, max_tokens: int
) -> Callable[[Iterable[Sentence]], Iterable[List[Sentence]]]:
    counter = token_counter(model)

    def batch_sentences(sentences: Iterable[Sentence]) -> Iterable[List[Sentence]]:
        batch: list[Sentence] = []
        count = 0
        for sentence in sentences:
            tokens = counter(sentence)
            if count + tokens > max_tokens:
                yield batch
                batch = [sentence]
                count = tokens
            else:
                batch.append(sentence)
                count += tokens

        if batch:
            yield batch

    return batch_sentences
