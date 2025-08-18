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

#include "render_particles.hpp"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

// includes for OpenGL
#include "helper_gl.hpp"

// includes
#include "helper_cuda.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>

namespace {
constexpr static auto fp64_colour = std::array{0.4f, 0.8f, 0.1f, 1.0f};
constexpr static auto fp32_colour = std::array{1.0f, 0.6f, 0.3f, 1.0f};
}    // namespace

ParticleRenderer::ParticleRenderer(std::size_t nb_bodies, float point_size, bool fp64) : colour_(nb_bodies * 4, 1.0f), m_spriteSize(point_size), m_bFp64Positions(fp64) {
    _initGL();
    reset(fp64, point_size);
}

auto ParticleRenderer::reset(bool fp64, float point_size) -> void {
    const auto& base_colour = fp64 ? fp64_colour : fp32_colour;

    setBaseColor(base_colour);
    setColours(colour_);
    setSpriteSize(point_size);
}

auto ParticleRenderer::reset(std::span<const float> colour, bool fp64, float point_size) -> void {
    const auto& base_colour = fp64 ? fp64_colour : fp32_colour;

    setBaseColor(base_colour);
    setColours(colour);
    setSpriteSize(point_size);
}

void ParticleRenderer::resetPBO() {
    // TODO: this function is never actually used?
    // TODO: glGenBuffers and glDeleteBuffers should be managed better
    glDeleteBuffers(1, reinterpret_cast<GLuint*>(&m_pbo));
}

void ParticleRenderer::set_positions(std::span<const float> pos) {
    assert(pos.size() == colour_.size());

    m_bFp64Positions = false;
    m_pos            = pos;

    if (!m_pbo) {
        glGenBuffers(1, reinterpret_cast<GLuint*>(&m_pbo));
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(float), pos.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    SDK_CHECK_ERROR_GL();
}
void ParticleRenderer::set_positions(std::span<const double> pos) {
    assert(pos.size() == colour_.size());

    m_bFp64Positions = true;
    m_pos_fp64       = pos;

    if (!m_pbo) {
        glGenBuffers(1, reinterpret_cast<GLuint*>(&m_pbo));
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(double), pos.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    SDK_CHECK_ERROR_GL();
}

void ParticleRenderer::setColours(std::span<const float> colour) {
    glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
    glBufferData(GL_ARRAY_BUFFER, colour.size() * sizeof(float), colour.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleRenderer::setPBO(unsigned int pbo, bool fp64) {
    m_pbo            = pbo;
    m_bFp64Positions = fp64;
}

void ParticleRenderer::_drawPoints(bool color) {
    const auto nb_particles = colour_.size() / 4;
    if (!m_pbo) {
        glBegin(GL_POINTS);
        {
            if (m_bFp64Positions) {
                for (auto i = 0; i < m_pos_fp64.size(); i += 4) {
                    glVertex3dv(&m_pos_fp64[i]);
                }
            } else {
                for (auto i = 0; i < nb_particles; i += 4) {
                    glVertex3fv(&m_pos[i]);
                }
            }
        }
        glEnd();
    } else {
        glEnableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, m_pbo);

        if (m_bFp64Positions) {
            glVertexPointer(4, GL_DOUBLE, 0, 0);
        } else {
            glVertexPointer(4, GL_FLOAT, 0, 0);
        }

        if (color) {
            glEnableClientState(GL_COLOR_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
            // glActiveTexture(GL_TEXTURE1);
            // glTexCoordPointer(4, GL_FLOAT, 0, 0);
            glColorPointer(4, GL_FLOAT, 0, 0);
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(nb_particles));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */) {
    switch (mode) {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            glUseProgram(m_programPoints);
            _drawPoints(false);
            glUseProgram(0);
            break;

        case PARTICLE_SPRITES:
        default:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(m_spriteSize);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(m_programSprites);
                GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, m_texture);

                glColor3f(1, 1, 1);
                glSecondaryColor3fv(m_baseColor.data());

                _drawPoints(false);

                glUseProgram(0);

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }

            break;

        case PARTICLE_SPRITES_COLOR:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(m_spriteSize);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(m_programSprites);
                GLuint texLoc = glGetUniformLocation(m_programSprites, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, m_texture);

                glColor3f(1, 1, 1);
                glSecondaryColor3fv(m_baseColor.data());

                _drawPoints(true);

                glUseProgram(0);

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }

            break;
    }

    SDK_CHECK_ERROR_GL();
}

void ParticleRenderer::_initGL() {
    constexpr static auto vertexShaderPoints =
        R"(void main() {
                vec4 vert = vec4(gl_Vertex.xyz, 1.0);
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;
                gl_FrontColor = gl_Color;
            })";

    constexpr static auto vertexShader =
        R"(void main() {
                float pointSize = 500.0 * gl_Point.size;
                vec4 vert = gl_Vertex;
                vert.w = 1.0;
                vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);
                gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));
                gl_TexCoord[0] = gl_MultiTexCoord0;
                gl_TexCoord[1] = gl_MultiTexCoord1;
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;
                gl_FrontColor = gl_Color;
                gl_FrontSecondaryColor = gl_SecondaryColor;
            })";

    constexpr static auto pixelShader =
        R"(uniform sampler2D splatTexture;
           void main() {
                vec4 color2 = gl_SecondaryColor;
                vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);
                gl_FragColor = color * color2;      // mix(vec4(0.1, 0.0, 0.0, color.w), color2, color.w);
            })";

    auto m_vertexShader       = glCreateShader(GL_VERTEX_SHADER);
    auto m_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
    auto m_pixelShader        = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(m_vertexShader, 1, &vertexShader, 0);
    glShaderSource(m_pixelShader, 1, &pixelShader, 0);
    glShaderSource(m_vertexShaderPoints, 1, &vertexShaderPoints, 0);

    glCompileShader(m_vertexShader);
    glCompileShader(m_vertexShaderPoints);
    glCompileShader(m_pixelShader);

    m_programSprites = glCreateProgram();
    glAttachShader(m_programSprites, m_vertexShader);
    glAttachShader(m_programSprites, m_pixelShader);
    glLinkProgram(m_programSprites);

    m_programPoints = glCreateProgram();
    glAttachShader(m_programPoints, m_vertexShaderPoints);
    glLinkProgram(m_programPoints);

    _createTexture();

    glGenBuffers(1, reinterpret_cast<GLuint*>(&m_vboColor));
    glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
    glBufferData(GL_ARRAY_BUFFER, colour_.size() * sizeof(float), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//------------------------------------------------------------------------------
// Function           : EvalHermite
// Description      :
//------------------------------------------------------------------------------
/**
 * EvalHermite(float pA, float pB, float vA, float vB, float u)
 * @brief Evaluates Hermite basis functions for the specified coefficients.
 */
constexpr auto evalHermite(float u) -> float {
    const auto u2 = u * u;
    const auto u3 = u2 * u;
    return 2 * u3 - 3 * u2 + 1;
}

template <std::size_t N> auto createGaussianMap() {
    constexpr auto Incr = 2.0f / N;

    auto M = std::array<float, 2 * N * N>{};
    auto B = std::array<unsigned char, 4 * N * N>{};
    auto i = 0;
    auto j = 0;

    // float mmax = 0;
    for (auto y = 0u; y < N; ++y) {
        const auto Y  = y * Incr - 1.0f;
        const auto Y2 = Y * Y;

        for (auto x = 0u; x < N; ++x, i += 2, j += 4) {
            const auto X     = x * Incr - 1.0f;
            const auto X2_Y2 = X * X + Y2;

            const auto dist = X2_Y2 > 1 ? 1.0f : std::sqrt(X2_Y2);

            M[i + 1] = M[i] = evalHermite(dist);
            B[j + 3] = B[j + 2] = B[j + 1] = B[j] = static_cast<unsigned char>(M[i] * 255);
        }
    }

    return B;
}

void ParticleRenderer::_createTexture() {
    constexpr auto resolution = 32;
    const auto     data       = createGaussianMap<resolution>();
    glGenTextures(1, reinterpret_cast<GLuint*>(&m_texture));
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
}
