#include "git_commit_id.hpp"

#include <iostream>

///
/// @brief Main function. Executable entry point. Currently no commandline arguments
///
/// @return int     Returns 0
///
auto main() -> int {

    std::cout << git_commit_id << '\n';

    return 0;
}
