#pragma once

#include <array>

class Camera {
 public:
    constexpr auto reset(const std::array<float, 3>& origin) noexcept -> void { translation_ = translation_lag_ = origin; }

    auto view_transform() noexcept -> void;

    auto zoom(float dy) noexcept -> void;

    constexpr auto translate(float dx, float dy) noexcept -> void {
        translation_[0] += dx / 100.0f;
        translation_[1] -= dy / 100.0f;
    }

    constexpr auto rotate(float dx, float dy) noexcept -> void {
        rotation_[0] += dy / 5.0f;
        rotation_[1] += dx / 5.0f;
    }

 private:
    std::array<float, 3> translation_lag_{0.f, -2.f, -150.f};
    std::array<float, 3> translation_{0.f, -2.f, -150.f};
    std::array<float, 3> rotation_{0.f, 0.f, 0.f};
};