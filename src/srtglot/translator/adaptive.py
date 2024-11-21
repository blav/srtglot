from functools import reduce
from typing import Callable, Type, TypeVar, List


T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=BaseException)

Mapper = Callable[[List[T]], List[U]]
FallbackMapper = Callable[[T, E], U]


def adaptive_map(
    input: List[T],
    mapper: Mapper[T, U],
    fallback: FallbackMapper[T, E, U],
    exception_type: Type[E],
) -> List[U]:
    state = [input]
    output: List[U] = []
    while state:
        try:
            head = state.pop(0)
            output.extend(mapper(head))
        except exception_type as e:
            if len(head) == 1:
                output.append(fallback(head[0], e))
                state = [reduce(lambda x, y: x + y, state)]
                continue

            mid = len(head) // 2
            state = [head[:mid], head[mid:]] + state

    return output
