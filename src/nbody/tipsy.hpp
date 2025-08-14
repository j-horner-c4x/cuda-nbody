#pragma once

#include <vector_types.h>

#include <array>
#include <filesystem>
#include <vector>

auto read_tipsy_file(const std::filesystem::path& fileName) -> std::array<std::vector<double>, 2>;

template <typename Real4> auto read_tipsy_file(const std::filesystem::path& fileName) -> std::array<std::vector<Real4>, 2>;

extern template auto read_tipsy_file<float4>(const std::filesystem::path& fileName) -> std::array<std::vector<float4>, 2>;
extern template auto read_tipsy_file<double4>(const std::filesystem::path& fileName) -> std::array<std::vector<double4>, 2>;
