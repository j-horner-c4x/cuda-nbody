#pragma once

#include <array>

struct CameraConfig {
    std::array<float, 3> translation_lag;
    std::array<float, 3> translation;
    std::array<float, 3> rotation;

    constexpr auto reset(const std::array<float, 3>& origin) noexcept -> void { translation = translation_lag = origin; }

    auto view_transform() noexcept -> void;
};