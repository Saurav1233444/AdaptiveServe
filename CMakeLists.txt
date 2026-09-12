cmake_minimum_required(VERSION 3.20)

project(AdaptiveServe)


set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(nlohmann_json REQUIRED)



include_directories(
    core
    workers/common
)



add_executable(

adaptive_server


core/gateway/main.cpp

core/gateway/server.cpp

core/gateway/request.cpp

core/gateway/response.cpp



core/router/router.cpp

core/router/scoring.cpp

core/router/policy.cpp



core/registry/model_info.cpp

core/registry/model_registry.cpp



core/scheduler/queue.cpp

core/scheduler/scheduler.cpp



core/monitor/latency.cpp

core/monitor/metrics.cpp

core/monitor/resource_monitor.cpp

core/monitor/history_logger.cpp

)



add_executable(

mobilenet_worker

workers/mobilenet/main.cpp

)



add_executable(

efficientnet_worker

workers/efficientnet/main.cpp

)



target_link_libraries(

adaptive_server

nlohmann_json::nlohmann_json

pthread

)



target_link_libraries(

mobilenet_worker

nlohmann_json::nlohmann_json

pthread

)



target_link_libraries(

efficientnet_worker

nlohmann_json::nlohmann_json

pthread

)
