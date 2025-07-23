#pragma once

#include <string>

namespace jh::library {

///
/// @brief The core implementation of the executable
///
/// This class makes up the library part of the executable, which means that the main logic is implemented here.
/// This kind of separation makes it easy to test the implementation for the executable, because the logic is nicely separated from the command-line logic implemented in the main function.
///
class LibraryClass {
 public:
    ///
    /// @brief Simply initializes the name member to the name of the project
    ///
    LibraryClass();

    ///
    /// @brief  A public method, a simple getter.
    ///
    /// @return string  The name of the project.
    ///
    [[nodiscard]] auto& get_name() const { return name_; }

 private:
    std::string name_;
};

}    // namespace jh::library
