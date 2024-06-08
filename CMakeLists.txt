cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

add_executable(main metric_learning_seq.cpp)
target_link_libraries(main ${TORCH_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 17)