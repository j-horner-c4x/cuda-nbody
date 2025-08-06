#pragma once

#include <vector_types.h>

#include <array>
#include <concepts>
#include <type_traits>

template <std::floating_point T> using vec3 = std::conditional_t<std::is_same_v<T, float>, float3, std::conditional_t<std::is_same_v<T, double>, double3, std::array<T, 3>>>;
template <std::floating_point T> using vec4 = std::conditional_t<std::is_same_v<T, float>, float4, std::conditional_t<std::is_same_v<T, double>, double4, std::array<T, 4>>>;