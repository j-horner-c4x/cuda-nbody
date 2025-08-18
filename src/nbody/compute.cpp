#include "compute.hpp"

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "camera.hpp"
#include "helper_cuda.hpp"
#include "render_particles.hpp"
#include "tipsy.hpp"

#include <chrono>
#include <print>

template <std::floating_point T> auto ComputeConfig::compare_results(BodySystemCUDA<T>& nbodyCuda) -> bool {
    bool passed = true;

    nbodyCuda.update(0.001f);

    {
        auto nbodyCpu = BodySystemCPU<T>(*this);

        nbodyCpu.set_position(nbodyCuda.get_position());
        nbodyCpu.set_velocity(nbodyCuda.get_velocity());

        nbodyCpu.update(0.001f);

        const auto cudaPos = nbodyCuda.get_position();
        const auto cpuPos  = nbodyCpu.get_position();

        constexpr auto tolerance = T{0.0005f};

        for (int i = 0; i < num_bodies_; i++) {
            if (std::abs(cpuPos[i] - cudaPos[i]) > tolerance) {
                passed = false;
                std::println("Error: (host){} != (device){}", cpuPos[i], cudaPos[i]);
            }
        }
    }
    if (passed) {
        std::println("  OK");
    }
    return passed;
}

ComputeConfig::ComputeConfig(
    bool                         enable_fp64,
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
    const std::filesystem::path& tipsy_file) {
    fp64_enabled_   = enable_fp64;
    cycle_demo_     = enable_cycle_demo;
    use_cpu_        = enable_cpu;
    compare_to_cpu_ = enable_compare_to_cpu;
    benchmark_      = enable_benchmark;
    use_host_mem_   = enable_host_memory;

    auto nb_devices_requested = 1;

    if (nb_requested_devices > 0) {
        nb_devices_requested = static_cast<int>(nb_requested_devices);
        std::println("number of CUDA devices  = {}", nb_devices_requested);
    }

    {
        auto nb_devices_available = 0;
        cudaGetDeviceCount(&nb_devices_available);

        if (nb_devices_available < nb_devices_requested) {
            throw std::invalid_argument(std::format("Error: only {} Devices available, {} requested.", nb_devices_available, nb_devices_requested));
        }
    }

    auto use_p2p = true;    // this is always optimal to use P2P path when available

    if (nb_devices_requested > 1) {
        // If user did not explicitly request host memory to be used, we default to P2P.
        // We fallback to host memory, if any of GPUs does not support P2P.
        if (!use_host_mem_) {
            auto all_gpus_support_p2p = true;
            // Enable P2P only in one direction, as every peer will access gpu0
            for (auto i = 1; i < nb_devices_requested; ++i) {
                auto canAccessPeer = 0;
                checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, i, 0));

                if (canAccessPeer != 1) {
                    all_gpus_support_p2p = false;
                }
            }

            if (!all_gpus_support_p2p) {
                use_host_mem_ = true;
                use_p2p       = false;
            }
        }
    }

    std::println("> Simulation data stored in {} memory", use_host_mem_ ? "system" : "video");
    std::println("> {} precision floating point simulation", fp64_enabled_ ? "Double" : "Single");
    std::println("> {} Devices used for simulation", nb_devices_requested);

    if (use_cpu_) {
        use_host_mem_   = true;
        compare_to_cpu_ = false;

#ifdef OPENMP
        std::println("> Simulation with CPU using OpenMP");
#else
        std::println("> Simulation with CPU");
#endif
    }

    auto dev_id = 0;

    auto custom_gpu = false;

    auto cuda_properties = cudaDeviceProp{};
    if (!use_cpu_) {
        if (device != -1) {
            custom_gpu = true;
        }

        // If the command-line has a device number specified, use it
        if (custom_gpu) {
            dev_id = device;
            assert(dev_id >= 0);

            const auto new_dev_ID = gpuDeviceInit(dev_id);

            if (new_dev_ID < 0) {
                throw std::invalid_argument(std::format("Could not use custom CUDA device: {}", dev_id));
            }

            dev_id = new_dev_ID;

        } else {
            // Otherwise pick the device with highest Gflops/s
            dev_id = gpuGetMaxGflopsDeviceId();
            checkCudaErrors(cudaSetDevice(dev_id));
            int major = 0, minor = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id));
            checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));
            std::println(R"(GPU Device {}: "{}" with compute capability {}.{}\n)", dev_id, _ConvertSMVer2ArchName(major, minor), major, minor);
        }

        checkCudaErrors(cudaGetDevice(&dev_id));
        checkCudaErrors(cudaGetDeviceProperties(&cuda_properties, dev_id));

        // Initialize devices
        assert(!(custom_gpu && (nb_devices_requested > 1)));

        if (custom_gpu || nb_devices_requested == 1) {
            auto properties = cudaDeviceProp{};
            checkCudaErrors(cudaGetDeviceProperties(&properties, dev_id));
            std::println("> Compute {}.{} CUDA device: [{}]", properties.major, properties.minor, properties.name);
            // CC 1.2 and earlier do not support double precision
            if (properties.major * 10 + properties.minor <= 12) {
                double_supported_ = false;
            }

        } else {
            for (int i = 0; i < nb_devices_requested; i++) {
                auto properties = cudaDeviceProp{};
                checkCudaErrors(cudaGetDeviceProperties(&properties, i));

                std::println("> Compute {}.{} CUDA device: [{}]", properties.major, properties.minor, properties.name);

                if (use_host_mem_) {
                    if (!properties.canMapHostMemory) {
                        throw std::invalid_argument(std::format("Device {} cannot map host memory!", i));
                    }

                    if (nb_devices_requested > 1) {
                        checkCudaErrors(cudaSetDevice(i));
                    }

                    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
                }

                // CC 1.2 and earlier do not support double precision
                if (properties.major * 10 + properties.minor <= 12) {
                    double_supported_ = false;
                }
            }
        }

        if (fp64_enabled_ && !double_supported_) {
            throw std::invalid_argument("One or more of the requested devices does not support double precision floating-point");
        }
    }

    nb_iterations_ = iterations == 0 ? 10 : static_cast<int>(iterations);
    auto blockSize = static_cast<int>(block_size);

    // default number of bodies is #SMs * 4 * CTA size
    if (use_cpu_) {
#ifdef OPENMP
        num_bodies = 8192;
#else
        num_bodies_ = 4096;
#endif
    } else if (nb_devices_requested == 1) {
        num_bodies_ = compare_to_cpu_ ? 4096 : blockSize * 4 * cuda_properties.multiProcessorCount;
    } else {
        num_bodies_ = 0;
        for (auto i = 0; i < nb_devices_requested; ++i) {
            auto properties = cudaDeviceProp{};
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));
            num_bodies_ += blockSize * (properties.major >= 2 ? 4 : 1) * properties.multiProcessorCount;
        }
    }

    if (nb_bodies != 0u) {
        num_bodies_ = static_cast<int>(nb_bodies);

        assert(num_bodies_ >= 1);

        if (num_bodies_ % blockSize) {
            auto new_nb_bodies = ((num_bodies_ / blockSize) + 1) * blockSize;
            std::println(R"(Warning: "number of bodies" specified {} is not a multiple of {}.)", num_bodies_, blockSize);
            std::println("Rounding up to the nearest multiple: {}.", new_nb_bodies);
            num_bodies_ = new_nb_bodies;
        } else {
            std::println("number of bodies = {}", num_bodies_);
        }
    }

    if (num_bodies_ <= 1024) {
        active_params_.m_clusterScale  = 1.52f;
        active_params_.m_velocityScale = 2.f;
    } else if (num_bodies_ <= 2048) {
        active_params_.m_clusterScale  = 1.56f;
        active_params_.m_velocityScale = 2.64f;
    } else if (num_bodies_ <= 4096) {
        active_params_.m_clusterScale  = 1.68f;
        active_params_.m_velocityScale = 2.98f;
    } else if (num_bodies_ <= 8192) {
        active_params_.m_clusterScale  = 1.98f;
        active_params_.m_velocityScale = 2.9f;
    } else if (num_bodies_ <= 16384) {
        active_params_.m_clusterScale  = 1.54f;
        active_params_.m_velocityScale = 8.f;
    } else if (num_bodies_ <= 32768) {
        active_params_.m_clusterScale  = 1.44f;
        active_params_.m_velocityScale = 11.f;
    }

    using enum NBodyConfig;

    if (!tipsy_file.empty()) {
        auto [positions, velocities] = read_tipsy_file(tipsy_file);

        tipsy_data_fp32_.positions.resize(positions.size());
        tipsy_data_fp32_.velocities.resize(velocities.size());

        using std::ranges::transform;

        constexpr auto to_float = [](double x) noexcept { return static_cast<float>(x); };

        transform(positions, tipsy_data_fp32_.positions.begin(), to_float);
        transform(velocities, tipsy_data_fp32_.velocities.begin(), to_float);

        tipsy_data_fp64_.positions  = std::move(positions);
        tipsy_data_fp64_.velocities = std::move(velocities);

        nbody_cpu_fp32_  = std::make_unique<BodySystemCPU<float>>(*this, tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities);
        nbody_cuda_fp32_ = std::make_unique<BodySystemCUDA<float>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities);

        if (double_supported_) {
            nbody_cpu_fp64_  = std::make_unique<BodySystemCPU<double>>(*this, tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            nbody_cuda_fp64_ = std::make_unique<BodySystemCUDA<double>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
        }
    } else {
        nbody_cpu_fp32_  = std::make_unique<BodySystemCPU<float>>(*this);
        nbody_cuda_fp32_ = std::make_unique<BodySystemCUDA<float>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id);

        if (double_supported_) {
            nbody_cpu_fp64_  = std::make_unique<BodySystemCPU<double>>(*this);
            nbody_cuda_fp64_ = std::make_unique<BodySystemCUDA<double>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id);
        }
    }

    if (use_cpu_) {
        reset_time_ = Clock::now();
    } else {
        checkCudaErrors(cudaEventCreate(&start_event_));
        checkCudaErrors(cudaEventCreate(&stop_event_));
        checkCudaErrors(cudaEventCreate(&host_mem_sync_event_));
    }

    demo_reset_time_ = Clock::now();
}

ComputeConfig ::~ComputeConfig() noexcept {
    if (!use_cpu_) {
        checkCudaErrors(cudaEventDestroy(start_event_));
        checkCudaErrors(cudaEventDestroy(stop_event_));
        checkCudaErrors(cudaEventDestroy(host_mem_sync_event_));
    }
}

template <typename BodySystemNew, typename BodySystemOld> auto ComputeConfig::switch_precision(BodySystemNew& new_nbody, BodySystemOld& old_nbody, ParticleRenderer& renderer) -> void {
    using T_new = BodySystemNew::Type;
    using T_old = BodySystemOld::Type;

    static_assert(!std::is_same_v<T_new, T_old>);

    cudaDeviceSynchronize();

    fp64_enabled_          = !fp64_enabled_;
    flops_per_interaction_ = fp64_enabled_ ? 30 : 20;

    const auto nb_bodies_4 = static_cast<std::size_t>(num_bodies_ * 4);

    auto oldPos = std::vector<T_old>(nb_bodies_4);
    auto oldVel = std::vector<T_old>(nb_bodies_4);

    using std::ranges::copy;
    copy(old_nbody.get_position(), oldPos.begin());
    copy(old_nbody.get_velocity(), oldVel.begin());

    // convert float to double
    auto newPos = std::vector<T_new>(nb_bodies_4);
    auto newVel = std::vector<T_new>(nb_bodies_4);

    for (int i = 0; i < nb_bodies_4; i++) {
        newPos[i] = static_cast<T_new>(oldPos[i]);
        newVel[i] = static_cast<T_new>(oldVel[i]);
    }

    new_nbody.set_position(newPos);
    new_nbody.set_velocity(newVel);

    renderer.reset(fp64_enabled_, active_params_.m_pointSize);

    cudaDeviceSynchronize();
}

template <typename BodySystem> auto ComputeConfig::run_benchmark(BodySystem& nbody) -> void {
    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    // once without timing to prime the device
    if (!use_cpu_) {
        nbody.update(active_params_.m_timestep);
    }

    auto milliseconds = 0.f;
    auto start        = TimePoint{};

    if (use_cpu_) {
        start = Clock::now();
    } else {
        checkCudaErrors(cudaEventRecord(start_event_, 0));
    }

    for (int i = 0; i < nb_iterations_; ++i) {
        nbody.update(active_params_.m_timestep);
    }

    if (use_cpu_) {
        milliseconds = MilliSeconds{Clock::now() - start}.count();
    } else {
        checkCudaErrors(cudaEventRecord(stop_event_, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event_));
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    }

    compute_perf_stats(milliseconds, nb_iterations_);

    std::println("{} bodies, total time for {} iterations: {:3} ms", num_bodies_, nb_iterations_, milliseconds);
    std::println("= {:3} billion interactions per second", interactions_per_second_);
    std::println("= {:3} {}-precision GFLOP/s at {} flops per interaction", g_flops_, std::is_same_v<typename BodySystem::Type, double> ? "double" : "single", flops_per_interaction_);
}

auto ComputeConfig::switch_precision(ParticleRenderer& renderer) -> void {
    if (double_supported_) {
        if (fp64_enabled_) {
            if (use_cpu_) {
                switch_precision(*nbody_cpu_fp32_, *nbody_cpu_fp64_, renderer);
            } else {
                switch_precision(*nbody_cuda_fp32_, *nbody_cuda_fp64_, renderer);
            }
            std::println("> Double precision floating point simulation");
        } else {
            if (use_cpu_) {
                switch_precision(*nbody_cpu_fp64_, *nbody_cpu_fp32_, renderer);
            } else {
                switch_precision(*nbody_cuda_fp64_, *nbody_cuda_fp32_, renderer);
            }
            std::println("> Single precision floating point simulation");
        }
    }
}

auto ComputeConfig::toggle_cycle_demo() -> void {
    cycle_demo_ = !cycle_demo_;
    std::println("Cycle Demo Parameters: {}\n", cycle_demo_ ? "ON" : "OFF");
}

auto ComputeConfig::select_demo(Camera& camera, ParticleRenderer& renderer) -> void {
    using enum NBodyConfig;

    select_demo();

    camera.reset(active_params_.camera_origin);

    if (tipsy_data_fp32_.positions.empty()) {
        if (use_cpu_) {
            if (fp64_enabled_) {
                nbody_cpu_fp64_->reset(*this, NBODY_CONFIG_SHELL, renderer.colour());
            } else {
                nbody_cpu_fp32_->reset(*this, NBODY_CONFIG_SHELL, renderer.colour());
            }
        } else {
            if (fp64_enabled_) {
                nbody_cuda_fp64_->reset(*this, NBODY_CONFIG_SHELL, renderer.colour());
            } else {
                nbody_cuda_fp32_->reset(*this, NBODY_CONFIG_SHELL, renderer.colour());
            }
        }
    } else {
        if (use_cpu_) {
            if (fp64_enabled_) {
                nbody_cpu_fp64_->set_position(tipsy_data_fp64_.positions);
                nbody_cpu_fp64_->set_velocity(tipsy_data_fp64_.velocities);
            } else {
                nbody_cpu_fp32_->set_position(tipsy_data_fp32_.positions);
                nbody_cpu_fp32_->set_velocity(tipsy_data_fp32_.velocities);
            }
        } else {
            if (fp64_enabled_) {
                nbody_cuda_fp64_->set_position(tipsy_data_fp64_.positions);
                nbody_cuda_fp64_->set_velocity(tipsy_data_fp64_.velocities);
            } else {
                nbody_cuda_fp32_->set_position(tipsy_data_fp32_.positions);
                nbody_cuda_fp32_->set_velocity(tipsy_data_fp32_.velocities);
            }
        }
    }
    demo_reset_time_ = Clock::now();

    renderer.reset(fp64_enabled_, active_params_.m_pointSize);
}

auto ComputeConfig::update_simulation(Camera& camera, ParticleRenderer& renderer) -> void {
    if (!paused_) {
        const auto demo_time = MilliSeconds{Clock::now() - demo_reset_time_}.count();

        if (cycle_demo_ && (demo_time > demoTime)) {
            next_demo(camera, renderer);
        }

        if (use_cpu_) {
            if (fp64_enabled_) {
                nbody_cpu_fp64_->update(active_params_.m_timestep);
            } else {
                nbody_cpu_fp32_->update(active_params_.m_timestep);
            }
        } else {
            if (fp64_enabled_) {
                nbody_cuda_fp64_->update(active_params_.m_timestep);
            } else {
                nbody_cuda_fp32_->update(active_params_.m_timestep);
            }
        }

        if (!use_cpu_) {
            cudaEventRecord(host_mem_sync_event_, 0);    // insert an event to wait on before rendering
        }
    }
}

auto ComputeConfig::display_NBody_system(ParticleRenderer::DisplayMode display_mode, ParticleRenderer& renderer) -> void {
    renderer.setSpriteSize(active_params_.m_pointSize);

    if (use_host_mem_) {
        if (use_cpu_) {
            if (fp64_enabled_) {
                renderer.set_positions(nbody_cpu_fp64_->get_position());
            } else {
                renderer.set_positions(nbody_cpu_fp32_->get_position());
            }
        } else {
            // This event sync is required because we are rendering from the host memory that CUDA is writing.
            // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
            cudaEventSynchronize(host_mem_sync_event_);

            if (fp64_enabled_) {
                renderer.set_positions(nbody_cuda_fp64_->get_position());
            } else {
                renderer.set_positions(nbody_cuda_fp32_->get_position());
            }
        }
    } else {
        assert(!use_cpu_);
        if (fp64_enabled_) {
            renderer.setPBO(nbody_cuda_fp64_->getCurrentReadBuffer(), num_bodies_, fp64_enabled_);
        } else {
            renderer.setPBO(nbody_cuda_fp32_->getCurrentReadBuffer(), num_bodies_, fp64_enabled_);
        }
    }

    // display particles
    renderer.display(display_mode);
}

auto ComputeConfig::reset(NBodyConfig initial_configuration, ParticleRenderer& renderer) -> void {
    if (tipsy_data_fp32_.positions.empty()) {
        if (use_cpu_) {
            if (fp64_enabled_) {
                nbody_cpu_fp64_->reset(*this, initial_configuration, renderer.colour());
            } else {
                nbody_cpu_fp32_->reset(*this, initial_configuration, renderer.colour());
            }
        } else {
            if (fp64_enabled_) {
                nbody_cuda_fp64_->reset(*this, initial_configuration, renderer.colour());
            } else {
                nbody_cuda_fp32_->reset(*this, initial_configuration, renderer.colour());
            }
        }
    } else {
        if (use_cpu_) {
            if (fp64_enabled_) {
                nbody_cpu_fp64_->set_position(tipsy_data_fp64_.positions);
                nbody_cpu_fp64_->set_velocity(tipsy_data_fp64_.velocities);
            } else {
                nbody_cpu_fp32_->set_position(tipsy_data_fp32_.positions);
                nbody_cpu_fp32_->set_velocity(tipsy_data_fp32_.velocities);
            }
        } else {
            if (fp64_enabled_) {
                nbody_cuda_fp64_->set_position(tipsy_data_fp64_.positions);
                nbody_cuda_fp64_->set_velocity(tipsy_data_fp64_.velocities);
            } else {
                nbody_cuda_fp32_->set_position(tipsy_data_fp32_.positions);
                nbody_cuda_fp32_->set_velocity(tipsy_data_fp32_.velocities);
            }
        }
    }

    renderer.reset(fp64_enabled_, active_params_.m_pointSize);
}

auto ComputeConfig::update_params() -> void {
    if (use_cpu_) {
        if (fp64_enabled_) {
            nbody_cpu_fp64_->update_params(active_params_);
        } else {
            nbody_cpu_fp32_->update_params(active_params_);
        }
    } else {
        if (fp64_enabled_) {
            nbody_cuda_fp64_->update_params(active_params_);
        } else {
            nbody_cuda_fp32_->update_params(active_params_);
        }
    }
}

auto ComputeConfig::get_milliseconds_passed() -> float {
    // stop timer
    if (use_cpu_) {
        const auto now          = Clock::now();
        const auto milliseconds = MilliSeconds{now - reset_time_}.count();

        reset_time_ = now;

        return milliseconds;
    }

    auto milliseconds = 0.f;

    checkCudaErrors(cudaEventRecord(stop_event_, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event_));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));

    return milliseconds;
}

auto ComputeConfig::restart_timer() -> void {
    // restart timer
    if (!use_cpu_) {
        checkCudaErrors(cudaEventRecord(start_event_, 0));
    }
}

auto ComputeConfig::calculate_fps(int fps_count) -> void {
    const auto milliseconds_passed = get_milliseconds_passed();
    restart_timer();

    const auto frequency = (1000.f / milliseconds_passed);
    fps_                 = static_cast<float>(fps_count) * frequency;

    compute_perf_stats(fps_);
}

auto ComputeConfig::run_benchmark() -> void {
    if (fp64_enabled_) {
        if (use_cpu_) {
            run_benchmark(*nbody_cpu_fp64_);
        } else {
            run_benchmark(*nbody_cuda_fp64_);
        }
    } else {
        if (use_cpu_) {
            run_benchmark(*nbody_cpu_fp32_);
        } else {
            run_benchmark(*nbody_cuda_fp32_);
        }
    }
}

auto ComputeConfig::compare_results() -> bool {
    return fp64_enabled_ ? compare_results(*nbody_cuda_fp64_) : compare_results(*nbody_cuda_fp32_);
}

template auto ComputeConfig::run_benchmark<BodySystemCPU<float>>(BodySystemCPU<float>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCPU<double>>(BodySystemCPU<double>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCUDA<float>>(BodySystemCUDA<float>& nbody) -> void;
template auto ComputeConfig::run_benchmark<BodySystemCUDA<double>>(BodySystemCUDA<double>& nbody) -> void;
