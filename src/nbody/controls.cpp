#include "controls.hpp"

#include "camera.hpp"
#include "compute.hpp"
#include "interface.hpp"
#include "render_particles.hpp"

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

auto ControlsConfig::mouse(int button, int state, int x, int y, InterfaceConfig& interface, ComputeConfig& compute) -> void {
    if (interface.show_sliders && interface.param_list->is_mouse_over(x, y)) {
        // call list mouse function
        interface.param_list->modify_sliders(x, y, button, state);
        compute.update_params();
    }

    set_state(button, state, x, y);

    glutPostRedisplay();
}

auto ControlsConfig::motion(int x, int y, InterfaceConfig& interface, CameraConfig& camera, ComputeConfig& compute) -> void {
    if (interface.show_sliders) {
        // call parameter list motion function
        if (interface.param_list->Motion(x, y)) {
            // by definition of this function, a mouse function is pressed so we need to update the parameters
            compute.update_params();
            glutPostRedisplay();
            return;
        }
    }

    move_camera(camera, x, y);

    glutPostRedisplay();
}

auto ControlsConfig::keyboard(unsigned char key, [[maybe_unused]] int x, [[maybe_unused]] int y, ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera, ParticleRenderer& renderer) -> void {
    using enum NBodyConfig;

    switch (key) {
        case ' ':
            compute.pause();
            break;

        case 27:    // escape
        case 'q':
        case 'Q':
            glutLeaveMainLoop();
            break;

        case 13:    // return
            compute.switch_precision(renderer);
            break;

        case '`':
            interface.toggle_sliders();
            break;

        case 'g':
        case 'G':
            interface.toggle_interactions();
            break;

        case 'p':
        case 'P':
            interface.cycle_display_mode();
            break;

        case 'c':
        case 'C':
            compute.toggle_cycle_demo();
            break;

        case '[':
            compute.previous_demo(camera, renderer);
            break;

        case ']':
            compute.next_demo(camera, renderer);
            break;

        case 'd':
        case 'D':
            interface.togle_display();
            break;

        case 'o':
        case 'O':
            compute.active_params.print();
            break;

        case '1':
            compute.reset<NBODY_CONFIG_SHELL>(renderer);
            break;

        case '2':
            compute.reset<NBODY_CONFIG_RANDOM>(renderer);
            break;

        case '3':
            compute.reset<NBODY_CONFIG_EXPAND>(renderer);
            break;
    }

    glutPostRedisplay();
}

auto ControlsConfig::special(int key, int x, int y, InterfaceConfig& interface) -> void {
    interface.param_list->Special(key, x, y);
    glutPostRedisplay();
}