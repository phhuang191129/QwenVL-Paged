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

enable_testing()

# Core allocator library implementing the headers under include/qwenvl_paged.
add_library(qwenvl_paged_core STATIC
    src/Block.cpp
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
    KVCacheManager
    Scheduler
    SwapBackend
)
foreach(spec IN LISTS qwenvl_spec_tests)
    add_executable(${spec}_test tests/${spec}.test.cpp)
    target_link_libraries(${spec}_test PRIVATE qwenvl_paged_core GTest::gtest_main)
    gtest_discover_tests(${spec}_test)
endforeach()
