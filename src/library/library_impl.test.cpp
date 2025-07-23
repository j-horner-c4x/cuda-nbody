#include "library_impl.hpp"

// 3rd party headers
#include <catch2/catch_test_macros.hpp>

namespace jh::library::test {

TEST_CASE("Library test") {
    const auto lib = LibraryClass{};

    REQUIRE(lib.get_name() == "cuda-nbody");
}

}    // namespace jh::library::test
