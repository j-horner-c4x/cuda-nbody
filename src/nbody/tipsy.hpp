#pragma once

#include <array>
#include <filesystem>
#include <vector>

auto read_tipsy_file(const std::filesystem::path& fileName) -> std::array<std::vector<double>, 2>;
