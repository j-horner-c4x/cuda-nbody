#include "interface.hpp"

#include "camera.hpp"
#include "compute.hpp"
#include "helper_gl.hpp"
#include "paramgl.hpp"
#include "win_coords.hpp"

#include <format>

auto InterfaceConfig::display(ComputeConfig& compute, CameraConfig& camera) -> void {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (display_enabled) {
        camera.view_transform();

        compute.display_NBody_system(display_mode);

        // display user interface
        if (show_sliders) {
            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            param_list->Render(0, 0);
            glDisable(GL_BLEND);
        }

        if (full_screen) {
            const auto win_coords = WinCoords{};

            constexpr static auto msg0 = std::string_view{"some_temp_device_name"};

            const auto msg1 = display_interactions ? std::format("{:.2} billion interactions per second", compute.interactions_per_second) : std::format("{:.2} GFLOP/s", compute.g_flops);

            const auto msg2 = std::format("{:.2} FPS [{} | {} bodies]", compute.fps, compute.fp64_enabled ? "double precision" : "single precision", compute.num_bodies);

            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            glColor3f(0.46f, 0.73f, 0.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 122, msg0, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 96, msg2, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 70, msg1, GLUT_BITMAP_TIMES_ROMAN_24);
            glDisable(GL_BLEND);
        }

        glutSwapBuffers();
    }
}