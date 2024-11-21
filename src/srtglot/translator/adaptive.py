from typing import Callable, Type, TypeVar, List


T = TypeVar("T")
U = TypeVar("U")

Mapper = Callable[[List[T]], List[U]]
FallbackMapper = Callable[[T], U]


def adaptive_map(
    input: List[T],
    mapper: Mapper[T, U],
    fallback: FallbackMapper[T, U],
    exception_type: Type[Exception],
) -> List[U]:
    state = [input]
    output: List[U] = []
    while state:
        try:
            head = state.pop(0)
            output.extend(mapper(head))
        except exception_type:
            if len(head) == 1:
                output.append(fallback(head[0]))
                continue

            mid = len(head) // 2
            state = [head[:mid], head[mid:]] + state

    return output
