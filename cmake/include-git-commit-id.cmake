#
# Git version
#
find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE "PROJECT_COMMIT_ID"
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git commit id: ${PROJECT_COMMIT_ID}")

    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE "PROJECT_TAG"
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git tag: ${PROJECT_TAG}")

    file(WRITE ./src/git_commit_id.hpp "constexpr static auto git_commit_id = \"${PROJECT_TAG} - ${PROJECT_COMMIT_ID}\";\n")
else(GIT_FOUND)
    file(WRITE ./src/git_commit_id.hpp "constexpr static auto git_commit_id = \"\";\n")
endif(GIT_FOUND)
