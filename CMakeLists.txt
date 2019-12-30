cmake_minimum_required(VERSION 3.10)
project(CNN C)

set(CMAKE_C_STANDARD 99)

add_executable(CNN src/main.c include/vector.h include/dataio.h include/network.h include/model.h include/memtool.h include/mytype.h)
add_executable(CNN_Old src/main_old.c include/vector.h include/dataio.h include/memtool.h include/mytype.h)
add_executable(CNN_OldOld src/main_oldold.c include/vector.h include/dataio.h)