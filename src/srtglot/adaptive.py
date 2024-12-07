from functools import reduce
from typing import TypeVar, Callable, Awaitable
from operator import add


T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=BaseException)

Mapper = Callable[[list[T]], Awaitable[list[U]]]
FallbackMapper = Callable[[T, E], Awaitable[U]]


async def adaptive_map(
    input: list[T],
    mapper: Mapper[T, U],
    fallback: FallbackMapper[T, E, U],
    exception_type: type[E],
) -> list[U]:
    state = [input]
    output: list[U] = []
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
