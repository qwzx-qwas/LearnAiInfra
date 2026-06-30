cmake_minimum_required(VERSION 3.20)

project(cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(math_lib SHARED
    src/add.cpp
)

target_include_directories(math_lib
    PUBLIC
    include
)

add_executable(cpp_demo
    src/main.cpp
)

target_link_libraries(cpp_demo
    PRIVATE
    math_lib
)
