#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::int64_t sum_list(
    const std::vector<std::int64_t>& numbers
) {
    return std::accumulate(
        numbers.begin(),
        numbers.end(),
        std::int64_t{0}
    );
}

std::int64_t sum_numpy(
    py::array_t<
        std::int64_t,
        py::array::c_style
    > array
) {
    py::buffer_info info = array.request();

    if (info.ndim != 1) {
        throw std::invalid_argument(
            "array must be one-dimensional"
        );
    }

    const auto* data =
        static_cast<const std::int64_t*>(info.ptr);

    const auto size =
        static_cast<std::size_t>(info.shape[0]);

    std::int64_t total = 0;

    for (std::size_t index = 0; index < size; ++index) {
        total += data[index];
    }

    return total;
}

PYBIND11_MODULE(cpp_ext, module) {
    module.doc() = "A minimal Python/C++ extension";

    module.def(
        "sum_list",
        &sum_list,
        py::arg("numbers")
    );

    module.def(
        "sum_numpy",
        &sum_numpy,
        py::arg("array")
    );
}
