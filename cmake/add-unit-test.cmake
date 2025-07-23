message("BUILD_TESTING: " ${BUILD_TESTING})
message("${PROJECT_NAME}_BUILD_TESTING: " ${${PROJECT_NAME}_BUILD_TESTING})
if(${PROJECT_NAME}_BUILD_TESTING)
    message("Adding ${PROJECT_NAME} tests.")
    include(CTest)

    #
    #   Catch2 unit testing framework
    #
    find_package(Catch2 CONFIG REQUIRED)
    include (Catch)
    include(ParseAndAddCatchTests)
    enable_testing()

    include(cmake/windows-set-path.cmake)

    #
    #   Unit test functions should have project specific prefixes, otherwise the functions may be overwritten when included as part of another project.
    #
    function(cuda_nbody_add_unit_test target)
        add_executable(${PROJECT_NAME}_${target} ${ARGN})
        target_link_libraries(${PROJECT_NAME}_${target} PRIVATE Catch2::Catch2 Catch2::Catch2WithMain)


        # This will automatically add the unit test so it can be used with CTests.
        # The "--durations yes" command line options get Catch2 to print out the test cases ran and their durations, otherwise it only reports failed tests.
        # The "--success" option reports every passed assertion which will be too verbose.
        catch_discover_tests(${PROJECT_NAME}_${target} EXTRA_ARGS --durations yes)
    endfunction()

    function(cuda_nbody_link_unit_test target)
        target_link_libraries(${PROJECT_NAME}_${target} PRIVATE ${ARGN})
    endfunction()
else()
    message("Not adding ${PROJECT_NAME} tests.")
    function(cuda_nbody_add_unit_test target)
    endfunction()

    function(cuda_nbody_link_unit_test target)
    endfunction()
endif()
