cmake_minimum_required(VERSION 3.20)

project(e57_pose_exporter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(E57Format CONFIG REQUIRED)

add_executable(e57_pose_exporter main.cpp)

target_link_libraries(e57_pose_exporter PRIVATE E57Format)