#include "camera.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define NOMINMAX
#include <GL/glew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#ifdef __linux__
#include <GL/glx.h>
#endif
#endif

auto CameraConfig::view_transform() noexcept -> void {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    static auto           camera_rot_lag = std::array{0.f, 0.f, 0.f};
    constexpr static auto inertia        = 0.1f;

    for (int c = 0; c < 3; ++c) {
        translation_lag[c] += (translation[c] - translation_lag[c]) * inertia;
        camera_rot_lag[c] += (rotation[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(translation_lag[0], translation_lag[1], translation_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
}
