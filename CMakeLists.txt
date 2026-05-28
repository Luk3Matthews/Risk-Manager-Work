cmake_minimum_required(VERSION 3.14)
project(active_passive CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Fetch xlnt automatically if not already installed ────────────────────────
include(FetchContent)

FetchContent_Declare(
    xlnt
    GIT_REPOSITORY https://github.com/tfussell/xlnt.git
    GIT_TAG        master
)

# Disable xlnt tests/samples to speed up build
set(XLNT_TESTS OFF CACHE BOOL "" FORCE)
set(XLNT_SAMPLES OFF CACHE BOOL "" FORCE)
set(XLNT_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(xlnt)

# ── Build the tool ──────────────────────────────────────────────────────────
add_executable(active_passive main.cpp)
target_link_libraries(active_passive PRIVATE xlnt)
