#include "git_commit_id.hpp"
#include "library/library_impl.hpp"

#include <iostream>

///
/// @brief Main function. Executable entry point. Currently no commandline arguments
///
/// @return int     Returns 0
///
auto main() -> int {
    const auto lib = jh::library::LibraryClass{};

    const auto message = "Hello from " + lib.get_name() + ", version: " + git_commit_id + " !";

    std::cout << message << '\n';

    return 0;
}
