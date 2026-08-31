cmake_minimum_required(VERSION 3.20)

project(
    ec_pro_native
    VERSION 0.1.0
    LANGUAGES C CXX
)

# llama.cpp at the pinned revision is safest with C++17.
# In C++20, u8"" literals become char8_t and break parts
# of llama-chat.cpp on MSVC.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# We only need the llama library itself.
set(
    LLAMA_BUILD_TESTS
    OFF
    CACHE BOOL ""
    FORCE
)

set(
    LLAMA_BUILD_TOOLS
    OFF
    CACHE BOOL ""
    FORCE
)

set(
    LLAMA_BUILD_EXAMPLES
    OFF
    CACHE BOOL ""
    FORCE
)

set(
    LLAMA_BUILD_SERVER
    OFF
    CACHE BOOL ""
    FORCE
)

set(
    LLAMA_BUILD_APP
    OFF
    CACHE BOOL ""
    FORCE
)

set(
    LLAMA_BUILD_COMMON
    OFF
    CACHE BOOL ""
    FORCE
)

# Build llama/ggml statically into the Python extension.
# Avoid needing separate llama.dll / ggml.dll files at runtime.
set(
    BUILD_SHARED_LIBS
    OFF
    CACHE BOOL ""
    FORCE
)

add_subdirectory(
    third_party/llama.cpp
)

find_package(
    Python
    COMPONENTS
        Interpreter
        Development.Module
    REQUIRED
)

find_package(
    pybind11
    CONFIG
    REQUIRED
)

pybind11_add_module(
    _ec_pro_native

    src/bindings.cpp
    src/token_processor.cpp
    src/inference_engine.cpp
    src/batch_scheduler.cpp
)

target_include_directories(
    _ec_pro_native
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    _ec_pro_native
    PRIVATE
        llama
)

if(MSVC)
    target_compile_options(
        _ec_pro_native
        PRIVATE
            /W4
            /EHsc
    )
else()
    target_compile_options(
        _ec_pro_native
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

set(
    EC_PRO_NATIVE_OUTPUT
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/llm_followups/native"
)

set_target_properties(
    _ec_pro_native
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${EC_PRO_NATIVE_OUTPUT}"

        RUNTIME_OUTPUT_DIRECTORY
            "${EC_PRO_NATIVE_OUTPUT}"
)

option(
    EC_PRO_NATIVE_BUILD_TESTS
    "Build EC Pro native tests"
    ON
)

if(EC_PRO_NATIVE_BUILD_TESTS)
    enable_testing()

    find_package(
        Threads
        REQUIRED
    )

    add_executable(
        ec_pro_native_scheduler_tests

        tests/test_batch_scheduler.cpp
        src/batch_scheduler.cpp
    )

    target_include_directories(
        ec_pro_native_scheduler_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ec_pro_native_scheduler_tests
        PRIVATE
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(
            ec_pro_native_scheduler_tests
            PRIVATE
                /W4
                /EHsc
        )
    else()
        target_compile_options(
            ec_pro_native_scheduler_tests
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

    add_test(
        NAME
            ec_pro_native_scheduler_tests
        COMMAND
            ec_pro_native_scheduler_tests
    )
endif()