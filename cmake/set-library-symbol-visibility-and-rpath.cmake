#
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
# The next two lines ensure that the shared library version doesn't export anything unintentionally.
# MSVC hides symbols by default, whereas GCC and Clang export everything.
# Exporting unintended symbols can cause conflicts and ODR violations as dependencies are added down the line, so libraries should always make their exports explicit (or at least use a linker script if retrofitting the code is too much).
# Still, if the user manually specifies a different setting, then we respect it.
#
if (NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# RPATH must be set before adding targets
# set RPATH: https://youtu.be/m0DwB4OvDXk?t=3119
# also see Professional CMake
include(GNUInstallDirs)
if(APPLE)
    set(base @loader_path)
else()
    set(base $ORIGIN)
endif()

file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH ${base} ${base}/${relDir})
