#
# Enable CUDA support if detected
#
cmake_policy(SET CMP0104 NEW)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA support detected")
    enable_language(CUDA)
    set(GPU 1)
    message(STATUS "CMAKE_CUDA_COMPILER_ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}")

    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "No CUDA support")
endif()
