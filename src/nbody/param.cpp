#include "param.hpp"

#include <format>

template <Numerical T> auto Param<T>::string() const -> std::string {
    return std::format("{:3f}", *ref_);
}

template auto Param<float>::string() const -> std::string;