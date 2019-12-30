cmake_minimum_required(VERSION 3.10)
project(C99_MNIST C)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_STANDARD 99)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)
add_executable(C99_MNIST src/main.c include/vector.h include/dataio.h include/memtool.h include/model.h include/network.h include/mytype.h)
add_executable(C99_MNIST_Test src/test.c include/vector.h include/dataio.h include/memtool.h include/model.h include/network.h include/mytype.h)
target_link_libraries(C99_MNIST m)
target_link_libraries(C99_MNIST_Test m)
