#
#   Static library - main implementation.
#   Consider keeping all implementation libraries static for simplicity.
#
add_library(${PROJECT_NAME}_render_particles STATIC render_particles.cpp)

add_library(${PROJECT_NAME}_bodysystemcuda STATIC bodysystemcuda.cu)

target_link_libraries(${PROJECT_NAME}_render_particles PRIVATE CUDA::cudart_static OpenGL::GL GLEW::GLEW)

target_link_libraries(${PROJECT_NAME}_bodysystemcuda
                            PRIVATE
                                $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)

#
# set_cuda_compile_options(<target>)
#
# Sets various compile and link options for the CUDA libraries
#
function(set_cuda_compile_options target)
    target_compile_features(${target} PUBLIC cxx_std_20)

    # p3.2xlarge
    # target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70 -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)

    # We need to explicitly state that we need all CUDA files in the
    # gpu library to be built with -dc as the member functions
    # could be called by other libraries and executables
    set_target_properties(${target} PROPERTIES  CUDA_SEPARABLE_COMPILATION ON
                                                POSITION_INDEPENDENT_CODE ON
                                                CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # link to cublas
    # target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::curand)
endfunction()

set_cuda_compile_options(${PROJECT_NAME}_bodysystemcuda)



#
#   Each unit test should correspond to one .cpp file and should be named with .test.
#   For more details see: https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.tests.merged
#   Unit test functions add ${PROJECT_NAME}_ prefix automatically.
#
# cuda_nbody_add_unit_test(library_impl_test library_impl.test.cpp)
# cuda_nbody_link_unit_test(library_impl_test PRIVATE ${PROJECT_NAME}_library_impl)
