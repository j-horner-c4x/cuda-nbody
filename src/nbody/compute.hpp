#pragma once

#include "bodysystem.hpp"
#include "params.hpp"
#include "render_particles.hpp"

#include <cuda_runtime.h>

#include <array>
#include <concepts>

#include <cassert>

struct CameraConfig;

struct ComputeConfig {
    bool        paused;
    bool        fp64_enabled;
    bool        cycle_demo;
    int         active_demo;
    bool        use_cpu;
    int         num_bodies;
    bool        double_supported;
    int         flops_per_interaction;
    bool        compare_to_cpu;
    bool        benchmark;
    bool        use_host_mem;
    float       g_flops;
    float       fps;
    float       interactions_per_second;
    NBodyParams active_params;
    cudaEvent_t host_mem_sync_event;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

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

    template <std::floating_point T_new, std::floating_point T_old> auto switch_demo_precision() -> void;

    template <typename BodySystem> auto run_benchmark(int iterations, BodySystem& nbody) -> void;

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
};

extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_EXPAND>() -> void;
extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_RANDOM>() -> void;
extern template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_SHELL>() -> void;

template <std::floating_point> class BodySystemCPU;
template <std::floating_point> class BodySystemCUDA;

extern template auto ComputeConfig::run_benchmark<BodySystemCPU<float>>(int iterations, BodySystemCPU<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCPU<double>>(int iterations, BodySystemCPU<double>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<float>>(int iterations, BodySystemCUDA<float>& nbody) -> void;
extern template auto ComputeConfig::run_benchmark<BodySystemCUDA<double>>(int iterations, BodySystemCUDA<double>& nbody) -> void;
