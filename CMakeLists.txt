cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_RayTracing CXX)
include_directories(./src)
add_executable(main src/main.cc)