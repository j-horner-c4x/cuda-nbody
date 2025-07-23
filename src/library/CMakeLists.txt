#
#   Static library - main implementation.
#   Consider keeping all implementation libraries static for simplicity.
#
# add_library(${PROJECT_NAME}_library_impl STATIC library_impl.cpp)

# target_link_libraries(${PROJECT_NAME}_library_impl PRIVATE fmt::fmt)

#
#   Each unit test should correspond to one .cpp file and should be named with .test.
#   For more details see: https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.tests.merged
#   Unit test functions add ${PROJECT_NAME}_ prefix automatically.
#
# cuda_nbody_add_unit_test(library_impl_test library_impl.test.cpp)
# cuda_nbody_link_unit_test(library_impl_test PRIVATE ${PROJECT_NAME}_library_impl)
