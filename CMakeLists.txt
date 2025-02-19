cmake_minimum_required(VERSION 3.15)
project(PSIM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the vcpkg toolchain file if not already set
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/gogoz/source/repos/GPU_PSIM/PSIM/stuff/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "" FORCE)
endif()

# Append vcpkg share directory to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "C:/Users/gogoz/source/repos/GPU_PSIM/PSIM/stuff/vcpkg/installed/x64-windows/share")

find_package(imgui CONFIG REQUIRED)

add_executable(PSIM main.cpp)
target_link_libraries(PSIM PRIVATE imgui::imgui)

target_include_directories(PSIM PRIVATE
  ${CMAKE_SOURCE_DIR}/stuff/vcpkg/installed/x64-windows/include
)
