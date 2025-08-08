#include "paramgl.hpp"

#include "helper_gl.hpp"
#include "win_coords.hpp"

#include <GL/freeglut.h>

namespace {

auto glPrintShadowed(int x, int y, std::string_view s, void* font, const float* color) -> void {
    glColor3f(0.0, 0.0, 0.0);
    glPrint(x - 1, y - 1, s, font);

    glColor3fv(reinterpret_cast<const GLfloat*>(color));
    glPrint(x, y, s, font);
}

}    // namespace

auto ParamListGL::AddParam(std::unique_ptr<ParamBase> param) -> void {
    m_params.push_back(std::move(param));
    auto& p = m_params.back();

    m_map[p->GetName()] = p.get();
    m_current           = m_params.begin();
}

auto ParamListGL::GetParam(std::string_view name) -> ParamBase& {
    const auto p_itr = m_map.find(name);

    assert(p_itr != m_map.end());

    return *(p_itr->second);
}

auto ParamListGL::ResetAll() -> void {
    for (auto& p : m_params) {
        p->Reset();
    }
}

ParamListGL::ParamListGL()
    : m_active(true), m_text_color_selected(1.0, 1.0, 1.0), m_text_color_unselected(0.75, 0.75, 0.75), m_text_color_shadow(0.0, 0.0, 0.0), m_bar_color_outer(0.25, 0.25, 0.25), m_bar_color_inner(1.0, 1.0, 1.0) {
    m_font       = (void*)GLUT_BITMAP_9_BY_15;    // GLUT_BITMAP_8_BY_13;
    m_font_h     = 15;
    m_bar_x      = 260;
    m_bar_w      = 250;
    m_bar_h      = 10;
    m_bar_offset = 5;
    m_text_x     = 5;
    m_separation = 15;
    m_value_x    = 200;
    m_start_x    = 0;
    m_start_y    = 0;
}

auto ParamListGL::Render(int x, int y, bool shadow) -> void {
    const auto win_coords = WinCoords{};

    m_start_x = x;
    m_start_y = y;

    for (auto p = m_params.begin(); p != m_params.end(); ++p) {
        if (p == m_current) {
            glColor3fv(&m_text_color_selected.r);
        } else {
            glColor3fv(&m_text_color_unselected.r);
        }

        if (shadow) {
            glPrintShadowed(x + m_text_x, y + m_font_h, (*p)->GetName(), m_font, (p == m_current) ? &m_text_color_selected.r : &m_text_color_unselected.r);
            glPrintShadowed(x + m_value_x, y + m_font_h, (*p)->GetValueString(), m_font, (p == m_current) ? &m_text_color_selected.r : &m_text_color_unselected.r);
        } else {
            glPrint(x + m_text_x, y + m_font_h, (*p)->GetName(), m_font);
            glPrint(x + m_value_x, y + m_font_h, (*p)->GetValueString(), m_font);
        }

        glColor3fv((GLfloat*)&m_bar_color_outer.r);
        glBegin(GL_LINE_LOOP);
        glVertex2f((GLfloat)(x + m_bar_x), (GLfloat)(y + m_bar_offset));
        glVertex2f((GLfloat)(x + m_bar_x + m_bar_w), (GLfloat)(y + m_bar_offset));
        glVertex2f((GLfloat)(x + m_bar_x + m_bar_w), (GLfloat)(y + m_bar_offset + m_bar_h));
        glVertex2f((GLfloat)(x + m_bar_x), (GLfloat)(y + m_bar_offset + m_bar_h));
        glEnd();

        glColor3fv((GLfloat*)&m_bar_color_inner.r);
        glRectf((GLfloat)(x + m_bar_x), (GLfloat)(y + m_bar_offset + m_bar_h), (GLfloat)(x + m_bar_x + ((m_bar_w - 1) * (*p)->GetPercentage())), (GLfloat)(y + m_bar_offset + 1));

        y += m_separation;
    }
}

auto ParamListGL::Mouse(int x, int y) -> bool {
    return Mouse(x, y, GLUT_LEFT_BUTTON, GLUT_DOWN);
}
auto ParamListGL::Mouse(int x, int y, int button) -> bool {
    return Mouse(x, y, button, GLUT_DOWN);
}

auto ParamListGL::Mouse(int x, int y, int button, int state) -> bool {
    if ((y < m_start_y) || (y > (int)(m_start_y + (m_separation * m_params.size()) - 1))) {
        m_active = false;
        return false;
    }

    m_active = true;

    int i = (y - m_start_y) / m_separation;

    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
#if defined(__GNUC__) && (__GNUC__ < 3)
        m_current = &m_params[i];
#else

        // MJH: workaround since the version of vector::at used here is
        // non-standard
        for (m_current = m_params.begin(); m_current != m_params.end() && i > 0; m_current++, i--)
            ;

        // m_current = (std::vector<ParamBase
        // *>::const_iterator)&m_params.at(i);
#endif

        if ((x > m_bar_x) && (x < m_bar_x + m_bar_w)) {
            Motion(x, y);
        }
    }

    return true;
}

auto ParamListGL::Motion(int x, int y) -> bool {
    if ((y < m_start_y) || (y > m_start_y + (m_separation * (int)m_params.size()) - 1)) {
        return false;
    }

    if (x < m_bar_x) {
        (*m_current)->SetPercentage(0.0);
        return true;
    }

    if (x > m_bar_x + m_bar_w) {
        (*m_current)->SetPercentage(1.0);
        return true;
    }

    (*m_current)->SetPercentage((x - m_bar_x) / (float)m_bar_w);
    return true;
}

auto ParamListGL::Special(int key, [[maybe_unused]] int x, [[maybe_unused]] int y) -> void {
    if (!m_active)
        return;

    switch (key) {
        case GLUT_KEY_DOWN:
            Increment();
            break;

        case GLUT_KEY_UP:
            Decrement();
            break;

        case GLUT_KEY_RIGHT:
            GetCurrent().Increment();
            break;

        case GLUT_KEY_LEFT:
            GetCurrent().Decrement();
            break;

        case GLUT_KEY_HOME:
            GetCurrent().Reset();
            break;

        case GLUT_KEY_END:
            GetCurrent().SetPercentage(1.0);
            break;
    }

    glutPostRedisplay();
}