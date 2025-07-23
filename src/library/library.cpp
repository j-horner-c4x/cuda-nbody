#include "../../include/library/library.hpp"

#include "library_impl.hpp"

#include <string>

namespace jh::library {

///
/// @brief  An simple function as an example library interface.
///
/// @return     string  "cuda-nbody"
///
auto example_function() -> std::string {
    auto lib = LibraryClass{};

    return lib.get_name();
}

}    // namespace jh::library
