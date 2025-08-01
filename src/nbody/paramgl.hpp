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

class ParamListGL : public ParamList {
 public:
    ParamListGL(const char* name = "");

    auto Render(int x, int y, bool shadow = false) -> void;

    auto Mouse(int x, int y) -> bool;
    auto Mouse(int x, int y, int button) -> bool;
    auto Mouse(int x, int y, int button, int state) -> bool;

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
