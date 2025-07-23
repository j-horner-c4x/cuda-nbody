#include "library_impl.hpp"

#include <fmt/format.h>

namespace jh::library {

LibraryClass::LibraryClass() : name_(fmt::format("{}-{}", "cuda", "nbody")) {}

}    // namespace jh::library
