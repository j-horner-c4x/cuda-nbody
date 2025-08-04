#include "win_coords.hpp"

#include <GL/freeglut.h>

WinCoords::WinCoords() noexcept {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(0.0f, static_cast<GLfloat>(glutGet(GLUT_WINDOW_HEIGHT) - 1), 0.0f);
    glScalef(1.0f, -1.0f, 1.0f);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);

    glMatrixMode(GL_MODELVIEW);
}

WinCoords::~WinCoords() noexcept {
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}