from __future__ import annotations


import cpp_ext

def main() -> None:
    result = cpp_ext.sum_list([1, 2, 3, 4, 5])

    print(f"Sum of the list is: {result}")
    assert result == 15

    keyword_result = cpp_ext.sum_list(
        numbers=[10, 20, 30, 40, 50]
    )
    assert keyword_result == 150

    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
