#pragma once

#include "gl_includes.hpp"

#include <GL/freeglut.h>

#include <string_view>

auto inline glPrint(int x, int y, std::string_view str, void* font) -> void {
    glRasterPos2f(static_cast<GLfloat>(x), static_cast<GLfloat>(y));
    for (const auto& c : str) {
        glutBitmapCharacter(font, c);
    }
}
