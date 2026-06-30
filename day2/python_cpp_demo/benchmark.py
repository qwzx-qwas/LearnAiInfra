from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable

import numpy as np
import cpp_ext

def python_sum_loop(numbers: list[int]) -> int:
    total = 0
    for number in numbers:
        total += number
    return total

def benchmark(
    name: str,
    function: Callable[[], object],
    *,
    number: int,
    repeat: int = 5
) -> None:
    samples = timeit.repeat(
        function,
        number=number,
        repeat=repeat
    )

    seconds_per_call = [
        sample / number
        for sample in samples
    ]


    best = min(seconds_per_call)
    median = statistics.median(seconds_per_call)

    print(
        f"{name:28s}"
        f"best={best * 1e6:10.3f} µs "
        f"median={median * 1e6:10.3f} µs"
    )

def run_case(size: int, number: int) -> None:
    python_list = list(range(size))
    numpy_array = np.asarray(
        python_list,
        dtype=np.int64
    )

    expected = size * (size - 1) // 2

    assert python_sum_loop(python_list) == expected
    assert cpp_ext.sum_list(python_list) == expected
    assert int(np.sum(numpy_array)) == expected

    print()
    print(f"size={size} calls={number}")

    benchmark(
        "pure Python loop",
        lambda: python_sum_loop(python_list),
        number=number,
    )


    benchmark(
        "C++ from Python list",
        lambda: cpp_ext.sum_list(python_list),
        number=number,
    )

    benchmark(
        "NumPy prebuilt array",
        lambda: np.sum(numpy_array),
        number=number,
    )

    benchmark(
        "NumPy including conversion",
        lambda: np.sum(
            np.array(
                python_list, dtype=np.int64
            )
        ),
        number=number,
    )

def main() -> None:
    run_case(size=10, number=100_000)
    run_case(size=1_000, number=10_000)
    run_case(size=100_000, number=100)

if __name__ == "__main__":
    main()
