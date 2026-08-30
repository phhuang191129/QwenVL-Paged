cmake_minimum_required(VERSION 3.16)
project(qwenvl_paged LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
endif()

include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

# JSONL trace parsing for the replay driver.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

enable_testing()

# Core allocator library implementing the headers under include/qwenvl_paged.
add_library(qwenvl_paged_core STATIC
    src/Block.cpp
    src/CacheLayout.cpp
    src/SwapBackend.cpp
    src/MemoryAllocator.cpp
    src/BlockTable.cpp
    src/KVCacheManager.cpp
    src/Scheduler.cpp
)
target_include_directories(qwenvl_paged_core PUBLIC include)

add_executable(memory_allocator_test tests/MemoryAllocator.test.cpp)
target_link_libraries(memory_allocator_test PRIVATE qwenvl_paged_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(memory_allocator_test)

# Specification (TDD) test targets written against the header contracts. The
# list maps a target prefix to its source file under tests/.
set(qwenvl_spec_tests
    Block
    BlockTable
    CacheView
    EndToEnd
    KVCacheManager
    PagedAttention
    Scheduler
    SwapBackend
    SynchronizationRules
)
foreach(spec IN LISTS qwenvl_spec_tests)
    add_executable(${spec}_test tests/${spec}.test.cpp)
    target_link_libraries(${spec}_test PRIVATE qwenvl_paged_core GTest::gtest_main)
    gtest_discover_tests(${spec}_test)
endforeach()

# Latency harness for allocator, fork, copy-on-write, and scheduler costs. Not
# registered with CTest: it reports timings rather than asserting on them.
add_executable(qwenvl_paged_bench bench/benchmark_main.cpp)
target_link_libraries(qwenvl_paged_bench PRIVATE qwenvl_paged_core)

# Qwen3-VL workload replay: drives the scheduler and cache manager from a trace
# and compares against a contiguous worst-case-reservation baseline.
add_library(qwenvl_trace_replay_core STATIC tools/trace_replay/TraceReplay.cpp)
target_include_directories(qwenvl_trace_replay_core PUBLIC tools/trace_replay)
target_link_libraries(qwenvl_trace_replay_core PUBLIC qwenvl_paged_core PRIVATE nlohmann_json::nlohmann_json)

add_executable(qwenvl_trace_replay tools/trace_replay/main.cpp)
target_link_libraries(qwenvl_trace_replay PRIVATE qwenvl_trace_replay_core)

# Golden PagedAttention fixtures shared by the CPU reference kernel and the
# future Triton kernel. Writes .npy arrays; pack_npz.py folds them into .npz.
add_executable(qwenvl_golden_export tools/golden_export/main.cpp)
target_link_libraries(qwenvl_golden_export PRIVATE qwenvl_paged_core)

# Python bindings, so the Triton kernel can read a pool this allocator manages
# instead of an exported snapshot. Off by default: the C++ build must not start
# depending on a Python toolchain.
option(QWENVL_BUILD_PYTHON "Build the pybind11 module" OFF)
if(QWENVL_BUILD_PYTHON)
    # FindPython rather than pybind11's legacy detection, which picks up the
    # system interpreter ahead of the uv-managed venv this project targets and
    # then fails on missing Python.h. Pass -DPython_EXECUTABLE=.venv/bin/python.
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    set_property(TARGET qwenvl_paged_core PROPERTY POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(qwenvl_paged_python python/bindings.cpp)
    target_link_libraries(qwenvl_paged_python PRIVATE qwenvl_paged_core)
    set_target_properties(qwenvl_paged_python PROPERTIES OUTPUT_NAME qwenvl_paged)
endif()

add_executable(TraceReplay_test tests/TraceReplay.test.cpp)
target_link_libraries(TraceReplay_test PRIVATE qwenvl_trace_replay_core GTest::gtest_main)
target_compile_definitions(TraceReplay_test PRIVATE QWENVL_TRACE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/traces")
gtest_discover_tests(TraceReplay_test)
