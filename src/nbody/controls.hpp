#pragma once

struct CameraConfig;

struct ControlsConfig {
    int button_state;
    int old_x;
    int old_y;

    auto set_state(int button, int state, int x, int y) noexcept -> void;

    auto move_camera(CameraConfig& camera, int x, int y) -> void;
};