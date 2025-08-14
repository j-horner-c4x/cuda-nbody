#pragma once

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "nbody_config.hpp"

#include <chrono>
#include <concepts>
#include <filesystem>
#include <memory>
#include <vector>

struct ComputeConfig;
struct NBodyParams;

template <typename BodySystem> class NBodyDemo {
 public:
    using PrecisionType = typename BodySystem::Type;

    NBodyDemo(std::filesystem::path tipsy_file) : tipsy_file_(std::move(tipsy_file)) {}

    template <typename = std::enable_if_t<BodySystem::use_cpu>> NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);

    template <typename = std::enable_if_t<!BodySystem::use_cpu>> NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);

    void _reset(const ComputeConfig& compute, NBodyConfig config, std::span<float> colour);

    void _selectDemo(ComputeConfig& compute, std::span<float> colour);

    auto get_arrays(std::span<PrecisionType> pos, std::span<PrecisionType> vel) -> void;
    auto set_arrays(std::span<const PrecisionType> pos, std::span<const PrecisionType> vel) -> void;

    auto _get_demo_time() -> float;

    auto _get_milliseconds_passed() -> float;

    auto update_simulation(float dt) -> void;

    auto update_params(const NBodyParams& active_params) -> void;

    auto& _get_impl() noexcept { return *m_nbody; }

 private:
    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    std::unique_ptr<BodySystem> m_nbody;

    std::vector<PrecisionType> m_hPos;
    std::vector<PrecisionType> m_hVel;

    TimePoint demo_reset_time_;

    TimePoint reset_time_;

    std::filesystem::path tipsy_file_;
};

template <std::floating_point T> class BodySystemCPU;
template <std::floating_point T> class BodySystemCUDA;

extern template NBodyDemo<BodySystemCPU<float>>;
extern template NBodyDemo<BodySystemCPU<double>>;
extern template NBodyDemo<BodySystemCUDA<float>>;
extern template NBodyDemo<BodySystemCUDA<double>>;

extern template NBodyDemo<BodySystemCPU<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
extern template NBodyDemo<BodySystemCPU<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
extern template NBodyDemo<BodySystemCUDA<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);
extern template NBodyDemo<BodySystemCUDA<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);