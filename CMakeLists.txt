cmake_minimum_required(VERSION 3.20)
project(AdaptiveServe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/.deps/onnxruntime" CACHE PATH "ONNX Runtime SDK root")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(adaptiveserve_native STATIC
    core/inference/engine.cpp
    core/monitor/resource_monitor.cpp
    core/registry/model_registry.cpp
    core/router/rule_router.cpp
    core/scheduler/bounded_scheduler.cpp
    core/utils/json_logger.cpp)
set_target_properties(adaptiveserve_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(adaptiveserve_native PUBLIC "${CMAKE_SOURCE_DIR}")
target_link_libraries(adaptiveserve_native PUBLIC onnxruntime nlohmann_json::nlohmann_json Threads::Threads)
target_compile_options(adaptiveserve_native PRIVATE -Wall -Wextra -Wpedantic)

pybind11_add_module(adaptive_core MODULE core/python/module.cpp)
target_link_libraries(adaptive_core PRIVATE adaptiveserve_native)
set_target_properties(adaptive_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

add_executable(adaptive_worker workers/onnx/main.cpp)
target_link_libraries(adaptive_worker PRIVATE adaptiveserve_native)
set_target_properties(adaptive_worker PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

include(CTest)
if(BUILD_TESTING)
    foreach(test_name IN ITEMS registry router scheduler monitor engine)
        add_executable(test_${test_name} tests/test_${test_name}.cpp)
        target_link_libraries(test_${test_name} PRIVATE adaptiveserve_native)
        add_test(NAME ${test_name} COMMAND test_${test_name})
        set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endforeach()

    set(ADAPTIVESERVE_TEST_ONNX "" CACHE FILEPATH "Optional tiny ONNX model for real-runtime integration test")
    if(ADAPTIVESERVE_TEST_ONNX)
        add_executable(test_onnx_integration tests/test_onnx_integration.cpp)
        target_link_libraries(test_onnx_integration PRIVATE adaptiveserve_native)
        add_test(NAME onnx_integration COMMAND test_onnx_integration "${ADAPTIVESERVE_TEST_ONNX}")
    endif()
endif()
