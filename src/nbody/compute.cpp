#include "compute.hpp"

#include "camera.hpp"
#include "helper_cuda.hpp"
#include "nbody_demo.hpp"

#include <chrono>
#include <print>

template <std::floating_point T_new, std::floating_point T_old> auto ComputeConfig::switch_demo_precision() -> void {
    static_assert(!std::is_same_v<T_new, T_old>);

    cudaDeviceSynchronize();

    fp64_enabled          = !fp64_enabled;
    flops_per_interaction = fp64_enabled ? 30 : 20;

    const auto nb_bodies_4 = static_cast<std::size_t>(num_bodies * 4);

    auto oldPos = std::vector<T_old>(nb_bodies_4);
    auto oldVel = std::vector<T_old>(nb_bodies_4);

    if (use_cpu) {
        NBodyDemo<BodySystemCPU<T_old>>::getArrays(oldPos, oldVel);
    } else {
        NBodyDemo<BodySystemCUDA<T_old>>::getArrays(oldPos, oldVel);
    }

    // convert float to double
    auto newPos = std::vector<T_new>(nb_bodies_4);
    auto newVel = std::vector<T_new>(nb_bodies_4);

    for (int i = 0; i < nb_bodies_4; i++) {
        newPos[i] = static_cast<T_new>(oldPos[i]);
        newVel[i] = static_cast<T_new>(oldVel[i]);
    }

    if (use_cpu) {
        NBodyDemo<BodySystemCPU<T_new>>::setArrays(newPos, newVel, *this);
    } else {
        NBodyDemo<BodySystemCUDA<T_new>>::setArrays(newPos, newVel, *this);
    }

    cudaDeviceSynchronize();
}

template <typename BodySystem> auto ComputeConfig::run_benchmark(int iterations, BodySystem& nbody) -> void {
    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    // once without timing to prime the device
    if (!use_cpu) {
        nbody.update(active_params.m_timestep);
    }

    auto milliseconds = 0.f;
    auto start        = TimePoint{};

    if (use_cpu) {
        start = Clock::now();
    } else {
        checkCudaErrors(cudaEventRecord(start_event, 0));
    }

    for (int i = 0; i < iterations; ++i) {
        nbody.update(active_params.m_timestep);
    }

    if (use_cpu) {
        milliseconds = MilliSeconds{Clock::now() - start}.count();
    } else {
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    }

    compute_perf_stats(milliseconds, iterations);

    std::println("{} bodies, total time for {} iterations: {:3} ms", num_bodies, iterations, milliseconds);
    std::println("= {:3} billion interactions per second", interactions_per_second);
    std::println("= {:3} {}-precision GFLOP/s at {} flops per interaction", g_flops, std::is_same_v<typename BodySystem::Type, double> ? "double" : "single", flops_per_interaction);
}

auto ComputeConfig::switch_precision() -> void {
    if (double_supported) {
        if (fp64_enabled) {
            switch_demo_precision<float, double>();
            std::println("> Double precision floating point simulation");
        } else {
            switch_demo_precision<double, float>();
            std::println("> Single precision floating point simulation");
        }
    }
}

auto ComputeConfig::toggle_cycle_demo() -> void {
    cycle_demo = !cycle_demo;
    std::println("Cycle Demo Parameters: {}\n", cycle_demo ? "ON" : "OFF");
}

auto ComputeConfig::select_demo(CameraConfig& camera) -> void {
    if (use_cpu) {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCPU<double>>::selectDemo(*this, camera);
        } else {
            NBodyDemo<BodySystemCPU<float>>::selectDemo(*this, camera);
        }
    } else {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCUDA<double>>::selectDemo(*this, camera);
        } else {
            NBodyDemo<BodySystemCUDA<float>>::selectDemo(*this, camera);
        }
    }
}

auto ComputeConfig::update_simulation(CameraConfig& camera) -> void {
    if (!paused) {
        auto demo_time = 0.f;

        if (use_cpu) {
            demo_time = fp64_enabled ? NBodyDemo<BodySystemCPU<double>>::get_demo_time() : NBodyDemo<BodySystemCPU<float>>::get_demo_time();
        } else {
            demo_time = fp64_enabled ? NBodyDemo<BodySystemCUDA<double>>::get_demo_time() : NBodyDemo<BodySystemCUDA<float>>::get_demo_time();
        }

        if (cycle_demo && (demo_time > demoTime)) {
            next_demo(camera);
        }

        if (use_cpu) {
            if (fp64_enabled) {
                NBodyDemo<BodySystemCPU<double>>::updateSimulation(active_params.m_timestep);
            } else {
                NBodyDemo<BodySystemCPU<float>>::updateSimulation(active_params.m_timestep);
            }
        } else {
            if (fp64_enabled) {
                NBodyDemo<BodySystemCUDA<double>>::updateSimulation(active_params.m_timestep);
            } else {
                NBodyDemo<BodySystemCUDA<float>>::updateSimulation(active_params.m_timestep);
            }
        }

        if (!use_cpu) {
            cudaEventRecord(host_mem_sync_event, 0);    // insert an event to wait on before rendering
        }
    }
}

auto ComputeConfig::display_NBody_system(ParticleRenderer::DisplayMode display_mode) -> void {
    if (use_cpu) {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCPU<double>>::display(*this, display_mode);
        } else {
            NBodyDemo<BodySystemCPU<float>>::display(*this, display_mode);
        }
    } else {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCUDA<double>>::display(*this, display_mode);
        } else {
            NBodyDemo<BodySystemCUDA<float>>::display(*this, display_mode);
        }
    }
}

template <NBodyConfig InitialConfiguration> auto ComputeConfig::reset() -> void {
    if (fp64_enabled) {
        if (use_cpu) {
            NBodyDemo<BodySystemCPU<double>>::reset(*this, InitialConfiguration);
        } else {
            NBodyDemo<BodySystemCUDA<double>>::reset(*this, InitialConfiguration);
        }
    } else {
        if (use_cpu) {
            NBodyDemo<BodySystemCPU<float>>::reset(*this, InitialConfiguration);
        } else {
            NBodyDemo<BodySystemCUDA<float>>::reset(*this, InitialConfiguration);
        }
    }
}

auto ComputeConfig::update_params() -> void {
    if (use_cpu) {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCPU<double>>::updateParams(active_params);
        } else {
            NBodyDemo<BodySystemCPU<float>>::updateParams(active_params);
        }
    } else {
        if (fp64_enabled) {
            NBodyDemo<BodySystemCUDA<double>>::updateParams(active_params);
        } else {
            NBodyDemo<BodySystemCUDA<float>>::updateParams(active_params);
        }
    }
}

auto ComputeConfig::finalize() noexcept -> void {
    if (!use_cpu) {
        checkCudaErrors(cudaEventDestroy(start_event));
        checkCudaErrors(cudaEventDestroy(stop_event));
        checkCudaErrors(cudaEventDestroy(host_mem_sync_event));
    }

    NBodyDemo<BodySystemCPU<float>>::Destroy();
    NBodyDemo<BodySystemCUDA<float>>::Destroy();

    if (double_supported) {
        NBodyDemo<BodySystemCPU<double>>::Destroy();
        NBodyDemo<BodySystemCUDA<double>>::Destroy();
    }
}

auto ComputeConfig::get_milliseconds_passed() -> float {
    // stop timer
    if (use_cpu) {
        return fp64_enabled ? NBodyDemo<BodySystemCPU<double>>::get_milliseconds_passed() : NBodyDemo<BodySystemCPU<float>>::get_milliseconds_passed();
    }

    auto milliseconds = 0.f;

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    return milliseconds;
}

auto ComputeConfig::restart_timer() -> void {
    // restart timer
    if (!use_cpu) {
        checkCudaErrors(cudaEventRecord(start_event, 0));
    }
}

template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_EXPAND>() -> void;
template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_RANDOM>() -> void;
template auto ComputeConfig::reset<NBodyConfig::NBODY_CONFIG_SHELL>() -> void;

template auto ComputeConfig::run_benchmark<BodySystemCPU<float>>(int iterations, BodySystemCPU<float>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCPU<double>>(int iterations, BodySystemCPU<double>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCUDA<float>>(int iterations, BodySystemCUDA<float>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCUDA<double>>(int iterations, BodySystemCUDA<double>& nbody) -> void;