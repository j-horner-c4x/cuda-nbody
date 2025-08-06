#pragma once

#include "bodysystem.hpp"
#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "render_particles.hpp"

#include <chrono>
#include <concepts>
#include <filesystem>
#include <memory>
#include <vector>

struct ComputeConfig;
struct CameraConfig;
struct NBodyParams;

template <typename BodySystem> class NBodyDemo {
 public:
    using PrecisionType = typename BodySystem::Type;

    static void Create(const std::filesystem::path& tipsy_file);
    static void Destroy() noexcept;

    static void init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute);

    static void reset(ComputeConfig& compute, NBodyConfig config);

    static void selectDemo(ComputeConfig& compute, CameraConfig& camera);

    static void runBenchmark(int iterations, ComputeConfig& compute);

    static void updateParams(const NBodyParams& active_params);

    static void updateSimulation(float dt);

    static void display(const ComputeConfig& compute, ParticleRenderer::DisplayMode display_mode);

    static void getArrays(std::vector<PrecisionType>& pos, std::vector<PrecisionType>& vel);

    static void setArrays(const std::vector<PrecisionType>& pos, const std::vector<PrecisionType>& vel, const ComputeConfig& compute);

    static auto get_demo_time() -> float;

    static auto get_milliseconds_passed() -> float;

    static auto& get_impl() noexcept { return *(m_singleton->m_nbody); }

    NBodyDemo(std::filesystem::path tipsy_file) : tipsy_file_(std::move(tipsy_file)) {}

 private:
    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    static std::unique_ptr<NBodyDemo> m_singleton;

    std::unique_ptr<BodySystem> m_nbody;

    std::unique_ptr<ParticleRenderer> m_renderer;

    std::vector<PrecisionType> m_hPos;
    std::vector<PrecisionType> m_hVel;
    std::vector<float>         m_hColor;

    TimePoint demo_reset_time_;

    TimePoint reset_time_;

    std::filesystem::path tipsy_file_;

 private:
    void _init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute);

    void _reset(ComputeConfig& compute, NBodyConfig config);

    void _resetRenderer(float point_size);

    void _selectDemo(ComputeConfig& compute, CameraConfig& camera);
};

template <std::floating_point T> class BodySystemCPU;
template <std::floating_point T> class BodySystemCUDA;

extern template NBodyDemo<BodySystemCPU<float>>;
extern template NBodyDemo<BodySystemCPU<double>>;
extern template NBodyDemo<BodySystemCUDA<float>>;
extern template NBodyDemo<BodySystemCUDA<double>>;