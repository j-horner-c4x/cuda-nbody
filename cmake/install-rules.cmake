if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_INSTALL_INCLUDEDIR include/${PROJECT_NAME} CACHE PATH "")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# default value of CMAKE_INSTALL_DOCDIR can be incorrect if this project is included as another project
set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME})

# find_package(<package>) call for consumers to find this project
set(package ${PROJECT_NAME})

#
#   Installs the executable to bin/ or equivalent.
#   Various options here most likely have no effect for executables but it has been kept the same as library for simplicity.
#
install(TARGETS ${PROJECT_NAME}_exe
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME COMPONENT ${PROJECT_NAME}_Runtime
    LIBRARY COMPONENT ${PROJECT_NAME}_Runtime NAMELINK_COMPONENT ${PROJECT_NAME}_Development
    ARCHIVE COMPONENT ${PROJECT_NAME}_Development
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

#
#   if the external dependency is a shared library, this needs to be installed too
#
#   vcpkg installs dependencies by default as dynamic on Windows and static on Linux...
#   could override these defaults by adding a custom triplet: https://learn.microsoft.com/en-gb/vcpkg/users/examples/overlay-triplets-linux-dynamic
#   but it is probably better and more robust to handle both cases properly here
#
#   this only needs to be done if building the executable and not the library as other dependencies should be specified (and installed via vcpkg)
#
function(install_dependency target)
    get_target_property(target_type ${target} TYPE)
    if (target_type STREQUAL SHARED_LIBRARY)
        install(IMPORTED_RUNTIME_ARTIFACTS ${target})
    endif()
endfunction()

install_dependency(fmt::fmt)


write_basic_package_version_file("${package}ConfigVersion.cmake" COMPATIBILITY SameMajorVersion)

# Allow package maintainers to freely override the path for the configs
set(${PROJECT_NAME}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}" CACHE PATH "CMake package config location relative to the install prefix")
mark_as_advanced(${PROJECT_NAME}_INSTALL_CMAKEDIR)

#
#   Creates installers (e.g. .rpm, .deb) on various platforms.
#   Doesn't really work very well (apart from self-extracting archive) but should provide a starting point if people really need it.
#
if(PROJECT_IS_TOP_LEVEL)
    # generic cpack variables
    set(CPACK_PACKAGE_VENDOR JHorner)
    set(CPACK_PACKAGE_DESCRIPTION_FILE ${PROJECT_SOURCE_DIR}/docs/Description.txt)
    set(CPACK_RESOURCE_FILE_WELCOME ${PROJECT_SOURCE_DIR}/docs/Welcome.txt)
    set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/docs/License.txt)
    set(CPACK_RESOURCE_FILE_README ${PROJECT_SOURCE_DIR}/README.md)
    set(CPACK_PACKAGE_CONTACT "Jonathan Horner")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})       # default version is typically not suitable
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_VERBATIM_VARIABLES TRUE)                              # only false by default for legacy reasons

    #  various other cpack options that can be set, look into this more if actually required

    # archive options
    # set(CPACK_ARCHIVE_COMPONENT_INSTALL TRUE)

    # deb options
    # set(CPACK_DEB_COMPONENT_INSTALL TRUE)
    # set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
    # set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS TRUE)
    # set(CPACK_DEBIAN_CUDA_NBODY_RUNTIME_PACKAGE_NAME ${PROJECT_NAME})
    # set(CPACK_DEBIAN_CUDA_NBODY_RUNTIME_FILE_NAME DEB-DEFAULT)
    # set(CPACK_DEBIAN_CUDA_NBODY_DEVELOPMENT_PACKAGE_NAME ${PROJECT_NAME}-devel)
    # set(CPACK_DEBIAN_CUDA_NBODY_DEVELOPMENT_FILE_NAME DEB-DEFAULT)

    # rpm options
    # set(CPACK_RPM_COMPONENT_INSTALL TRUE)
    # set(CPACK_RPM_PACKAGE_ARCHITECTURE TRUE)
    # set(CPACK_RPM_DEBUGINFO_PACKAGE TRUE)
    # set(CPACK_RPM_CUDA_NBODY_RUNTIME_PACKAGE_NAME ${PROJECT_NAME})
    # set(CPACK_RPM_CUDA_NBODY_RUNTIME_FILE_NAME RPM-DEFAULT)
    # set(CPACK_RPM_CUDA_NBODY_DEVELOPMENT_PACKAGE_NAME ${PROJECT_NAME}-devel)
    # set(CPACK_RPM_CUDA_NBODY_DEVELOPMENT_FILE_NAME RPM-DEFAULT)

    # to create install packages call cpack in the build directory e.g.:
    # $ cd <build_dir>/
    # $ cpack
    if(WIN32)
        set(CPACK_GENERATOR ZIP WIX)
    else()
        # set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

        if(APPLE)
            set(CPACK_GENERATOR TGZ STGZ productbuild)
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            # detecting exact linux flavour seems to be a massive pain so if other distros are need the appropriate generators can either be added here or ran directly on the commandline e.g.
            # $ cpack -G "RPM"
            set(CPACK_GENERATOR TGZ STGZ DEB)
        else()
            set(CPACK_GENERATOR TGZ STGZ)
        endif()
    endif()

    #
    #   Allows development stuff to be an optional install component.
    #
    include(CPack)

    cpack_add_component(${PROJECT_NAME}_Runtime
                        DISPLAY_NAME Runtime
                        DESCRIPTION "Shared libraries and executables"
                        REQUIRED
                        INSTALL_TYPES Minimal)

    cpack_add_component(${PROJECT_NAME}_Development
                        DISPLAY_NAME "Developer pre-requisites"
                        DESCRIPTION "Headers/static libs needed for building"
                        DEPENDS ${PROJECT_NAME}_Runtime
                        INSTALL_TYPES Developer)

    cpack_add_install_type(Minimal)
    cpack_add_install_type(Developer DISPLAY_NAME "Application/library Development")
endif()

message("CMAKE_INSTALL_FULL_BINDIR: " ${CMAKE_INSTALL_FULL_BINDIR})
message("${PROJECT_NAME}_INSTALL_CMAKEDIR: " ${${PROJECT_NAME}_INSTALL_CMAKEDIR})
