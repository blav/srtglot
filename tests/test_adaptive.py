import pytest
from srtglot.adaptive import adaptive_map
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_adaptive_map():
    mapper = AsyncMock()
    mapper.side_effect = lambda x: [int(720 / x) for x in x]

    fallback = AsyncMock()
    fallback.side_effect = lambda x, e: 0

    assert await adaptive_map(
        [1, 2, 3, 0, 4, 5, 6], mapper, fallback, ZeroDivisionError
    ) == [
        720,
        360,
        240,
        0,
        180,
        144,
        120,
    ]

    assert mapper.call_count == 6
    assert mapper.call_args_list[0] == (([1, 2, 3, 0, 4, 5, 6],),)
    assert mapper.call_args_list[1] == (([1, 2, 3],),)
    assert mapper.call_args_list[2] == (([0, 4, 5, 6],),)
    assert mapper.call_args_list[3] == (([0, 4],),)
    assert mapper.call_args_list[4] == (([0],),)
    assert fallback.call_count == 1
    assert fallback.call_args[0][0] == 0
    assert (
        fallback.call_args[0][1].__repr__() == "ZeroDivisionError('division by zero')"
    )
    assert mapper.call_args_list[5] == (([4, 5, 6],),)
