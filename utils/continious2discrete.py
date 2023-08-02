import numpy as np
import numpy.typing as npt


def continious2discrete(
    continious_from: float, continious_to: float, discrete_chunks: int
):
    discrete_space = []
    chunk_margin = abs(continious_to - continious_from) / discrete_chunks
    chunk_from = continious_from
    for _ in range(discrete_chunks):
        chunk_to = chunk_from + chunk_margin
        discrete_space.append((chunk_from, chunk_to))
        chunk_from = chunk_to
    return discrete_space


def search4discrete(discrete_space: list, continious_observation: float):
    lo, hi = 0, len(discrete_space) - 1
    while lo <= hi:
        mid = lo + round((hi - lo) / 2)
        observation_limit_from, observation_limit_to = discrete_space[mid]
        if continious_observation > observation_limit_to:
            lo = mid + 1
        elif continious_observation < observation_limit_from:
            hi = mid - 1
        else:
            return mid
    return -1
