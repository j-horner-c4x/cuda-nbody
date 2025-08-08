#include "controls.hpp"

#include "camera.hpp"

#include <GL/freeglut.h>

auto ControlsConfig::set_state(int button, int state, int x, int y) noexcept -> void {
    if (state == GLUT_DOWN) {
        button_state |= 1 << button;
    } else if (state == GLUT_UP) {
        button_state = 0;
    }

    const auto mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        button_state = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        button_state = 3;
    }

    old_x = x;
    old_y = y;
}

auto ControlsConfig::move_camera(CameraConfig& camera, int x, int y) -> void {
    const auto dx = static_cast<float>(x - old_x);
    const auto dy = static_cast<float>(y - old_y);

    if (button_state == 3) {
        // left+middle = zoom
        camera.translation[2] += (dy / 100.0f) * 0.5f * std::abs(camera.translation[2]);
    } else if (button_state & 2) {
        // middle = translate
        camera.translation[0] += dx / 100.0f;
        camera.translation[1] -= dy / 100.0f;
    } else if (button_state & 1) {
        // left = rotate
        camera.rotation[0] += dy / 5.0f;
        camera.rotation[1] += dx / 5.0f;
    }

    old_x = x;
    old_y = y;
}