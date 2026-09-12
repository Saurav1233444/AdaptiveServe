add_executable(
mobilenet_worker

workers/mobilenet/main.cpp

)


add_executable(
efficientnet_worker

workers/efficientnet/main.cpp

)


target_link_libraries(
mobilenet_worker
pthread
)


target_link_libraries(
efficientnet_worker
pthread
)
