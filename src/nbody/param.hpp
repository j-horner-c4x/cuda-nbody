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

#pragma once

#include <concepts>
#include <string>
#include <type_traits>

#include <cassert>

// base class for named parameter
class ParamBase {
 public:
    ParamBase(std::string name) noexcept : name_(std::move(name)) {}
    virtual ~ParamBase() noexcept = default;

    auto& name() const noexcept { return name_; }

    auto virtual string() const -> std::string = 0;

    auto virtual reset() const noexcept -> void = 0;
    auto virtual increment() const -> void      = 0;
    auto virtual decrement() const -> void      = 0;

    auto virtual percentage() const noexcept -> float           = 0;
    auto virtual set_percentage(float p) const noexcept -> void = 0;

 protected:
    std::string name_;
};

template <typename T>
concept Numerical = std::floating_point<T> || std::integral<T>;

// derived class for single-valued parameter
template <Numerical T> class Param final : public ParamBase {
 public:
    static_assert(std::is_same_v<T, float>, "only Param<float> has been implemented so far");

    Param(std::string name, T value, T min, T max, T step, T* ptr) : ParamBase(std::move(name)), ref_(ptr), default_(value), min_(min), max_(max), step_(step) {
        assert(ref_);
        *ref_ = value;
    }
    ~Param() = default;

    auto string() const -> std::string override;

    auto percentage() const noexcept -> float override { return (*ref_ - min_) / static_cast<float>(max_ - min_); }

    auto set_percentage(float p) const noexcept -> void override { *ref_ = static_cast<T>(min_ + p * (max_ - min_)); }

    auto reset() const noexcept -> void override { *ref_ = default_; }

    auto increment() const noexcept -> void override {
        *ref_ += step_;

        if (*ref_ > max_) {
            *ref_ = max_;
        }
    }

    auto decrement() const noexcept -> void override {
        *ref_ -= step_;

        if (*ref_ < min_) {
            *ref_ = min_;
        }
    }

 private:
    T* ref_;    // pointer to value declared elsewhere
    T  default_;
    T  min_;
    T  max_;
    T  step_;
};

extern template auto Param<float>::string() const -> std::string;