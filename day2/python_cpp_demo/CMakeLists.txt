cmake_minimum_required(VERSION 3.20)


project(python_cpp_demo LANGUAGES CXX)

find_package(Python 
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_ext
    cpp/binding.cpp
)

target_compile_features(cpp_ext
    PRIVATE
    cxx_std_17
)
