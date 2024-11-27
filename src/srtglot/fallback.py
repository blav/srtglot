from abc import abstractmethod
from .languages import LanguageTokenization


class Strategy:
    @abstractmethod
    def join(self, left: str, right: str) -> str:
        pass

    @abstractmethod
    def split(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def can_split(self, text: str) -> bool:
        pass


class SpaceStrategy(Strategy):
    def join(self, left: str, right: str) -> str:
        return f"{left} {right}"

    def split(self, text: str) -> list[str]:
        return text.rsplit(" ", 1)

    def can_split(self, text: str) -> bool:
        return " " in text


class CJKStrategy(Strategy):
    def join(self, left: str, right: str) -> str:
        return f"{left}{right}"

    def split(self, text: str) -> list[str]:
        middle = len(text) // 2
        return [text[:middle], text[middle:]]

    def can_split(self, text: str) -> bool:
        return len(text) > 1


def fit_fragments_count(
    tokenization: LanguageTokenization,
    source: int,
    target: list[str],
) -> list[str]:
    strategy = (
        SpaceStrategy() if tokenization == LanguageTokenization.SPACE else CJKStrategy()
    )

    def shrink(target: list[str]) -> list[str]:
        if len(target) == source:
            return target

        target = target.copy()
        index = min(range(len(target)), key=lambda i: len(target[i]))
        if index > 0:
            target[index - 1] = strategy.join(target[index - 1], target[index])
            target.pop(index)
        else:
            target[index] = strategy.join(target[index], target[index + 1])
            target.pop(index + 1)

        return shrink(target)

    def expand(target: list[str]) -> list[str]:
        if len(target) == source:
            return target

        target = target.copy()
        split_indices = sorted(
            filter(lambda t: strategy.can_split(t[1]), enumerate(target)),
            key=lambda x: -len(x[1]),
        )

        if not split_indices:
            return target + [""]

        index = split_indices[0][0]
        target = (
            target[:index]
            + strategy.split(target[index])
            + target[index + 1:]
        )

        return expand(target)

    delta = len(target) - source
    if delta == 0:
        return target
    elif delta > 0:
        return shrink(target)
    else:
        return expand(target)
