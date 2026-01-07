cmake_minimum_required(VERSION 3.0)
project(final_project)

find_package(OpenGL REQUIRED)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(BUILD_SHARED_LIBS OFF)

add_subdirectory(external)

include_directories(
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glad-opengl-3.3/include/
	external/tinygltf-2.9.3/
	final_project/
)

add_executable(final_project
	final_project/final_project_main.cpp
	final_project/render/shader.cpp)
target_link_libraries(final_project
	${OPENGL_LIBRARY}
	glfw
	glad
)