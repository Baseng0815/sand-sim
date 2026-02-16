cmake_minimum_required(VERSION 3.16)
project(sand-sim CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Collect all sources and headers from src/
file(GLOB_RECURSE SOURCES "src/main.cu" "src/simulation.cu" "src/transition_table.cpp")
# file(GLOB_RECURSE HEADERS "")

add_executable(sand-sim ${SOURCES} ${HEADERS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL3 REQUIRED sdl3)
pkg_check_modules(SDL3_TTF REQUIRED sdl3-ttf)
pkg_check_modules(CUDA REQUIRED cuda)

target_link_libraries(sand-sim PRIVATE ${SDL3_LIBRARIES} ${SDL3_TTF_LIBRARIES} ${CUDA_LIBRARIES} m)
target_include_directories(sand-sim PRIVATE ${SDL3_INCLUDE_DIRS} ${SDL3_TTF_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_compile_options(sand-sim PRIVATE ${SDL3_CFLAGS_OTHER} ${SDL3_TTF_CFLAGS_OTHER} ${CUDA_CFLAGS_OTHER})

# Generate compile_commands.json for clangd/LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
