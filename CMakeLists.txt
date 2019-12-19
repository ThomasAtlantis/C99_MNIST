cmake_minimum_required(VERSION 3.10)
project(CNN C)

set(CMAKE_C_STANDARD 99)

add_executable(CNN main.c vector.h)

add_executable(CNN_TEST test.c)