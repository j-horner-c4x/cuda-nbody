/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
   ParamListGL
   - class derived from ParamList to do simple OpenGL rendering of a parameter
   list sgg 8/2001
*/

#pragma once

#include "param.hpp"

#include <map>
#include <memory>
#include <string_view>
#include <vector>

#include <cassert>

class ParamListGL {
 public:
    ParamListGL();

    auto AddParam(std::unique_ptr<ParamBase> param) -> void;

    auto Render(int x, int y, bool shadow = false) -> void;

    auto is_mouse_over(int x, int y) noexcept -> bool;
    auto modify_sliders(int x, int y, int button, int state) -> void;

    auto Motion(int x, int y) -> bool;

    auto Special(int key, [[maybe_unused]] int x, [[maybe_unused]] int y) -> void;

    auto SetFont(void* font, int height) -> void {
        m_font   = font;
        m_font_h = height;
    }

    void SetSelectedColor(float r, float g, float b) { m_text_color_selected = Color{r, g, b}; }
    void SetUnSelectedColor(float r, float g, float b) { m_text_color_unselected = Color{r, g, b}; }
    void SetBarColorInner(float r, float g, float b) { m_bar_color_inner = Color{r, g, b}; }
    void SetBarColorOuter(float r, float g, float b) { m_bar_color_outer = Color{r, g, b}; }

    void SetActive(bool b) { m_active = b; }

 private:
    // look-up parameter based on name
    auto GetParam(std::string_view name) -> ParamBase&;

    auto& GetParam(std::size_t i) noexcept {
        assert(i < m_params.size());
        return *(m_params[i]);
    }

    auto& GetCurrent() noexcept {
        assert(m_current != m_params.end());
        return **(m_current);
    }

    auto GetSize() const noexcept { return m_params.size(); }

    // functions to traverse list
    auto Reset() noexcept -> void { m_current = m_params.begin(); }

    auto Increment() noexcept {
        ++m_current;

        if (m_current == m_params.end()) {
            m_current = m_params.begin();
        }
    }

    auto Decrement() noexcept {
        if (m_current == m_params.begin()) {
            m_current = m_params.end() - 1;
        } else {
            m_current--;
        }
    }

    auto ResetAll() -> void;

    std::vector<std::unique_ptr<ParamBase>>                 m_params;
    std::map<std::string, ParamBase*, std::less<>>          m_map;
    std::vector<std::unique_ptr<ParamBase>>::const_iterator m_current;

    void* m_font;
    int   m_font_h;    // font height

    int m_bar_x;         // bar start x position
    int m_bar_w;         // bar width
    int m_bar_h;         // bar height
    int m_text_x;        // text start x position
    int m_separation;    // bar separation in y
    int m_value_x;       // value text x position
    int m_bar_offset;    // bar offset in y

    int m_start_x, m_start_y;

    bool m_active;

    struct Color {
        float r, g, b;
    };

    Color m_text_color_selected;
    Color m_text_color_unselected;
    Color m_text_color_shadow;
    Color m_bar_color_outer;
    Color m_bar_color_inner;
};
