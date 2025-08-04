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
 Simple parameter system
 sgreen@nvidia.com 4/2001
*/

#pragma once

#include <string>
#include <type_traits>

#include <cassert>

// base class for named parameter
class ParamBase {
 public:
    ParamBase(std::string name) noexcept : m_name(std::move(name)) {}
    virtual ~ParamBase() noexcept = default;

    auto& GetName() const noexcept { return m_name; }

    auto virtual GetValueString() const noexcept -> std::string = 0;

    virtual void Reset()     = 0;
    virtual void Increment() = 0;
    virtual void Decrement() = 0;

    virtual float GetPercentage()        = 0;
    virtual void  SetPercentage(float p) = 0;

 protected:
    std::string m_name;
};

// derived class for single-valued parameter
template <class T> class Param final : public ParamBase {
 public:
    static_assert(std::is_same_v<T, float>, "only Param<float> has been implemented so far");

    Param(std::string name, T value, T min, T max, T step, T* ptr) : ParamBase(std::move(name)), m_ptr(ptr), m_default(value), m_min(min), m_max(max), m_step(step) {
        assert(m_ptr);
        *m_ptr = value;
    }
    ~Param() = default;

    auto GetValueString() const noexcept -> std::string override;

    float GetPercentage() override { return (*m_ptr - m_min) / (float)(m_max - m_min); }

    void SetPercentage(float p) override { *m_ptr = (T)(m_min + p * (m_max - m_min)); }

    void Reset() override { *m_ptr = m_default; }

    void Increment() override {
        *m_ptr += m_step;

        if (*m_ptr > m_max) {
            *m_ptr = m_max;
        }
    }

    void Decrement() override {
        *m_ptr -= m_step;

        if (*m_ptr < m_min) {
            *m_ptr = m_min;
        }
    }

 private:
    T* m_ptr;    // pointer to value declared elsewhere
    T  m_default;
    T  m_min;
    T  m_max;
    T  m_step;
};

extern template auto Param<float>::GetValueString() const noexcept -> std::string;