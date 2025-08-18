#pragma once

#include "nbody_config.hpp"
#include "params.hpp"
#include "render_particles.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <concepts>
#include <filesystem>
#include <memory>

#include <cassert>

class Camera;
class ParticleRenderer;

template <typename BodySystem> class NBodyDemo;
template <std::floating_point T> class BodySystemCPU;
template <std::floating_point T> class BodySystemCUDA;

struct ComputeConfig {
    ComputeConfig(bool                         enable_fp64,
                  bool                         enable_cycle_demo,
                  bool                         enable_cpu,
                  bool                         enable_compare_to_cpu,
                  bool                         enable_benchmark,
                  bool                         enable_host_memory,
                  int                          device,
                  std::size_t                  nb_requested_devices,
                  std::size_t                  block_size,
                  std::size_t                  nb_bodies,
                  const std::filesystem::path& tipsy_file);

    auto nb_bodies() const noexcept { return num_bodies_; }

    auto& active_params() const noexcept { return active_params_; }

    auto interactions_per_second() const noexcept { return interactions_per_second_; }

    auto gflops() const noexcept { return g_flops_; }

    auto fps() const noexcept { return fps_; }

    auto fp64_enabled() const noexcept { return fp64_enabled_; }

    auto paused() const noexcept { return paused_; }

    auto use_pbo() const noexcept { return use_pbo_; }

    auto use_host_mem() const noexcept { return use_host_mem_; }

    auto create_sliders() -> ParamListGL { return active_params_.create_sliders(); }

    auto run_benchmark(int nb_iterations) -> void;

    auto compare_results() -> bool;

    auto pause() noexcept -> void { paused_ = !paused_; }

    auto switch_precision(ParticleRenderer& renderer) -> void;

    auto toggle_cycle_demo() -> void;

    auto previous_demo(Camera& camera, ParticleRenderer& renderer) -> void;

    auto next_demo(Camera& camera, ParticleRenderer& renderer) -> void;

    auto update_simulation(Camera& camera, ParticleRenderer& renderer) -> void;

    auto display_NBody_system(ParticleRenderer::DisplayMode display_mode, ParticleRenderer& renderer) -> void;

    auto reset(NBodyConfig initial_configuration, ParticleRenderer& renderer) -> void;

    auto update_params() -> void;

    auto calculate_fps(int frame_count) -> void;

    ~ComputeConfig() noexcept;

 private:
    template <typename BodySystem> auto run_benchmark(int nb_iterations, BodySystem& nbody) -> void;

    template <typename BodySystemNew, typename BodySystemOld> auto switch_precision(BodySystemNew& new_nbody, BodySystemOld& old_nbody, ParticleRenderer& renderer) -> void;

    template <std::floating_point T> auto compare_results(BodySystemCUDA<T>& nbodyCuda) const -> bool;

    auto select_demo(Camera& camera, ParticleRenderer& renderer) -> void;

    constexpr auto compute_perf_stats(float frequency) -> void;

    auto get_milliseconds_passed() -> float;

    constexpr static auto demoParams = std::array{
        NBodyParams{0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
        NBodyParams{0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
        NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
        NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
        NBodyParams{0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
        NBodyParams{0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
        NBodyParams{0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0, -50}};

    constexpr static auto numDemos = demoParams.size();

    constexpr static auto demoTime = 10000.0f;    // ms

    bool        paused_ = false;
    bool        fp64_enabled_;
    bool        cycle_demo_;
    int         active_demo_ = 0;
    bool        use_cpu_;
    int         num_bodies_       = 16384;
    bool        double_supported_ = true;
    bool        use_host_mem_;
    bool        use_pbo_;
    float       g_flops_                 = 0.f;
    float       fps_                     = 0.f;
    float       interactions_per_second_ = 0.f;
    NBodyParams active_params_           = demoParams[0];
    cudaEvent_t host_mem_sync_event_{};
    cudaEvent_t start_event_{};
    cudaEvent_t stop_event_{};

    std::unique_ptr<BodySystemCPU<float>>  nbody_cpu_fp32_;
    std::unique_ptr<BodySystemCUDA<float>> nbody_cuda_fp32_;

    std::unique_ptr<BodySystemCPU<double>>  nbody_cpu_fp64_;
    std::unique_ptr<BodySystemCUDA<double>> nbody_cuda_fp64_;

    template <std::floating_point T> struct TipsyData {
        std::vector<T> positions;
        std::vector<T> velocities;
    };

    TipsyData<float>  tipsy_data_fp32_;
    TipsyData<double> tipsy_data_fp64_;

    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    TimePoint demo_reset_time_;

    TimePoint reset_time_;
};

template <std::floating_point> class BodySystemCPU;
template <std::floating_point> class BodySystemCUDA;

extern template auto ComputeConfig::run_benchmark<BodySystemCPU<float>>(int nb_iterations, BodySystemCPU<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCPU<double>>(int nb_iterations, BodySystemCPU<double>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<float>>(int nb_iterations, BodySystemCUDA<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<double>>(int nb_iterations, BodySystemCUDA<double>& nbody) -> void;
