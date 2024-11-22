from functools import reduce
from typing import Callable, Type, TypeVar, List, Awaitable
from operator import add


T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=BaseException)

Mapper = Callable[[List[T]], Awaitable[List[U]]]
FallbackMapper = Callable[[T, E], Awaitable[U]]


async def adaptive_map(
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
            output.extend(await mapper(head))
        except exception_type as e:
            if len(head) == 1:
                output.append(await fallback(head[0], e))
                state = [reduce(add, state)] if state else []
                continue

            mid = len(head) // 2
            state = [head[:mid], head[mid:]] + state

    return output
