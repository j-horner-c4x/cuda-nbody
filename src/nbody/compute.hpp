#pragma once

#include "nbody_config.hpp"
#include "params.hpp"
#include "render_particles.hpp"

#include <cuda_runtime.h>

#include <array>
#include <concepts>
#include <filesystem>
#include <memory>

#include <cassert>

struct CameraConfig;

template <typename BodySystem> class NBodyDemo;
template <std::floating_point T> class BodySystemCPU;
template <std::floating_point T> class BodySystemCUDA;

struct ComputeConfig {
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

    bool        paused = false;
    bool        fp64_enabled;
    bool        cycle_demo;
    int         active_demo = 0;
    bool        use_cpu;
    int         num_bodies            = 16384;
    bool        double_supported      = true;
    int         flops_per_interaction = fp64_enabled ? 30 : 20;
    bool        compare_to_cpu;
    bool        benchmark;
    int         nb_iterations;
    bool        use_host_mem;
    float       g_flops                 = 0.f;
    float       fps                     = 0.f;
    float       interactions_per_second = 0.f;
    NBodyParams active_params           = demoParams[0];
    cudaEvent_t host_mem_sync_event{};
    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    ComputeConfig(bool                         enable_fp64,
                  bool                         enable_cycle_demo,
                  bool                         enable_cpu,
                  bool                         enable_compare_to_cpu,
                  bool                         enable_benchmark,
                  bool                         enable_host_memory,
                  int                          device,
                  std::size_t                  nb_requested_devices,
                  std::size_t                  iterations,
                  std::size_t                  block_size,
                  std::size_t                  nb_bodies,
                  const std::filesystem::path& tipsy_file);

    template <typename BodySystem> auto run_benchmark(BodySystem& nbody) -> void;

    auto run_benchmark() -> void;

    auto compare_results() -> bool;

    auto select_demo() -> void {
        assert(active_demo < numDemos);

        active_params = demoParams[active_demo];
    }

    auto finalize() noexcept -> void;

    auto pause() noexcept -> void { paused = !paused; }

    auto switch_precision() -> void;

    auto toggle_cycle_demo() -> void;

    auto previous_demo(CameraConfig& camera) -> void {
        active_demo = (active_demo == 0) ? numDemos - 1 : (active_demo - 1) % numDemos;
        select_demo(camera);
    }

    auto next_demo(CameraConfig& camera) -> void {
        active_demo = (active_demo + 1) % numDemos;
        select_demo(camera);
    }

    auto select_demo(CameraConfig& camera) -> void;

    auto update_simulation(CameraConfig& camera) -> void;

    auto display_NBody_system(ParticleRenderer::DisplayMode display_mode) -> void;

    template <NBodyConfig InitialConfiguration> auto reset() -> void;

    auto update_params() -> void;

    constexpr auto compute_perf_stats(float frequency) -> void {
        // double precision uses intrinsic operation followed by refinement, resulting in higher operation count per interaction.
        // Note: Astrophysicists use 38 flops per interaction no matter what, based on "historical precedent", but they are using FLOP/s as a measure of "science throughput".
        // We are using it as a measure of hardware throughput.  They should really use interactions/s...
        interactions_per_second = (static_cast<float>(num_bodies * num_bodies) * 1e-9f) * frequency;

        g_flops = interactions_per_second * static_cast<float>(flops_per_interaction);
    }

    constexpr auto compute_perf_stats() -> void { compute_perf_stats(fps); }

    constexpr auto compute_perf_stats(float milliseconds, int iterations) -> void { compute_perf_stats(iterations * (1000.0f / milliseconds)); }

    auto get_milliseconds_passed() -> float;

    auto restart_timer() -> void;

    auto calculate_fps(int fps_count) -> void;

    ~ComputeConfig() noexcept;

 private:
    template <typename BodySystemNew, typename BodySystemOld> auto switch_precision(BodySystemNew& new_nbody, BodySystemOld& old_nbody) -> void;

    std::unique_ptr<NBodyDemo<BodySystemCPU<float>>>  nbody_cpu_fp32;
    std::unique_ptr<NBodyDemo<BodySystemCUDA<float>>> nbody_cuda_fp32;

    std::unique_ptr<NBodyDemo<BodySystemCPU<double>>>  nbody_cpu_fp64;
    std::unique_ptr<NBodyDemo<BodySystemCUDA<double>>> nbody_cuda_fp64;
};

extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_EXPAND>() -> void;
extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_RANDOM>() -> void;
extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_SHELL>() -> void;

template <std::floating_point> class BodySystemCPU;
template <std::floating_point> class BodySystemCUDA;

extern template auto ComputeConfig::run_benchmark<BodySystemCPU<float>>(BodySystemCPU<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCPU<double>>(BodySystemCPU<double>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<float>>(BodySystemCUDA<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<double>>(BodySystemCUDA<double>& nbody) -> void;
