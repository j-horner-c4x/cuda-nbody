#pragma once

class WinCoords {
 public:
    /// @brief calls begin_win_coords
    WinCoords() noexcept;

    WinCoords(const WinCoords&) = delete;
    WinCoords(WinCoords&&)      = delete;

    auto operator=(const WinCoords&) = delete;
    auto operator=(WinCoords&&)      = delete;

    /// @brief calls end_win_coords
    ~WinCoords() noexcept;
};
