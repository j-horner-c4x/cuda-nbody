include("${CMAKE_CURRENT_LIST_DIR}/cuda-nbodyTargets.cmake")

# if built as a static library, the consumer also needs its internal dependencies installed as well
# this should be handled automatically if installed by vcpkg
