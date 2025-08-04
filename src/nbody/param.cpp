#include "param.hpp"

#include <format>

template <typename T> auto Param<T>::GetValueString() const noexcept -> std::string {
    return std::format("{:3f}", *m_ptr);
}

template auto Param<float>::GetValueString() const noexcept -> std::string;