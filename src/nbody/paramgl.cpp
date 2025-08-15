#include "paramgl.hpp"

#include "helper_gl.hpp"
#include "win_coords.hpp"

#include <GL/freeglut.h>

namespace {

// constexpr auto font_ = GLUT_BITMAP_9_BY_15;    // GLUT_BITMAP_8_BY_13;

constexpr static auto font_h_ = 15;    // font height

constexpr static auto bar_x_      = 280;    // bar start x position
constexpr static auto bar_w_      = 250;    // bar width
constexpr static auto bar_h_      = 10;     // bar height
constexpr static auto text_x_     = 5;      // text start x position
constexpr static auto separation_ = 15;     // bar separation in y
constexpr static auto value_x_    = 200;    // value text x position
constexpr static auto bar_offset_ = 5;      // bar offset in y

struct Colour {
    float r, g, b;
};

constexpr static auto text_color_selected_   = Colour{1.f, 1.f, 1.f};
constexpr static auto text_color_unselected_ = Colour{0.75f, 0.75f, 0.75f};
constexpr static auto text_color_shadow_     = Colour{0.f, 0.f, 0.f};
constexpr static auto bar_color_outer_       = Colour{0.25f, 0.25f, 0.25f};
constexpr static auto bar_color_inner_       = Colour{0.8f, 0.8f, 0.f};

}    // namespace

auto ParamListGL::add_param(std::unique_ptr<ParamBase> param) -> void {
    params_.push_back(std::move(param));
    current_ = params_.begin();
}

auto ParamListGL::render() const -> void {
    const auto win_coords = WinCoords{};

    constexpr auto set_colour = [](const Colour& colour) { glColor3fv(reinterpret_cast<const GLfloat*>(&colour.r)); };

    constexpr auto x = 0;
    auto           y = 0;

    constexpr auto x_begin = static_cast<GLfloat>(x + bar_x_);
    constexpr auto x_end   = static_cast<GLfloat>(x + bar_x_ + bar_w_);

    const auto current = current_->get();

    for (const auto& p : params_) {
        if (p.get() == current) {
            set_colour(text_color_selected_);
        } else {
            set_colour(text_color_unselected_);
        }

// can't make font constexpr aparently...
#define FONT GLUT_BITMAP_9_BY_15
        glPrint(x + text_x_, y + font_h_, p->name(), FONT);
        glPrint(x + value_x_, y + font_h_, p->string(), FONT);
#undef FONT

        set_colour(bar_color_outer_);

        const auto y_begin = static_cast<GLfloat>(y + bar_offset_);
        const auto y_end   = static_cast<GLfloat>(y + bar_offset_ + bar_h_);

        glBegin(GL_LINE_LOOP);
        glVertex2f(x_begin, y_begin);
        glVertex2f(x_end, y_begin);
        glVertex2f(x_end, y_end);
        glVertex2f(x_begin, y_end);
        glEnd();

        set_colour(bar_color_inner_);
        glRectf(x_begin, y_end, static_cast<GLfloat>(x + bar_x_ + ((bar_w_ - 1) * p->percentage())), static_cast<GLfloat>(y + bar_offset_ + 1));

        y += separation_;
    }
}

auto ParamListGL::is_mouse_over([[maybe_unused]] int x, int y) noexcept -> bool {
    active_ = (y >= 0) && (y <= static_cast<int>((separation_ * params_.size()) - 1));

    return active_;
}

auto ParamListGL::modify_sliders(int button, int state, int x, int y) -> void {
    assert(active_);

    const auto i = y / separation_;

    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
        current_ = params_.begin() + i;

        if ((x > bar_x_) && (x < bar_x_ + bar_w_)) {
            motion(x, y);
        }
    }
}

auto ParamListGL::motion(int x, int y) const -> bool {
    if ((y < 0) || (y > (separation_ * params_.size()) - 1)) {
        return false;
    }

    if (x < bar_x_) {
        (*current_)->set_percentage(0.0);
        return true;
    }

    if (x > bar_x_ + bar_w_) {
        (*current_)->set_percentage(1.0);
        return true;
    }

    (*current_)->set_percentage((x - bar_x_) / static_cast<float>(bar_w_));
    return true;
}

auto ParamListGL::special(int key, [[maybe_unused]] int x, [[maybe_unused]] int y) -> void {
    if (!active_)
        return;

    switch (key) {
        case GLUT_KEY_DOWN:
            ++current_;

            if (current_ == params_.end()) {
                current_ = params_.begin();
            }
            break;

        case GLUT_KEY_UP:
            if (current_ == params_.begin()) {
                current_ = params_.end() - 1;
            } else {
                --current_;
            }
            break;

        case GLUT_KEY_RIGHT:
            (*current_)->increment();
            break;

        case GLUT_KEY_LEFT:
            (*current_)->decrement();
            break;

        case GLUT_KEY_HOME:
            (*current_)->reset();
            break;

        case GLUT_KEY_END:
            (*current_)->set_percentage(1.0);
            break;
    }

    glutPostRedisplay();
}