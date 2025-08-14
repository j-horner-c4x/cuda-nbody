#include "nbody_demo.hpp"

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "camera.hpp"
#include "compute.hpp"
#include "helper_cuda.hpp"
#include "randomise_bodies.hpp"

template <typename BodySystem> template <typename> NBodyDemo<BodySystem>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config) : tipsy_file_(std::move(tipsy_file)) {
    m_nbody = std::make_unique<BodySystem>(compute.num_bodies);

    const auto nb_bodies_4 = compute.num_bodies * 4;

    // allocate host memory
    m_hPos.resize(nb_bodies_4);
    m_hVel.resize(nb_bodies_4);
    m_hColor.resize(nb_bodies_4);

    m_nbody->setSoftening(compute.active_params.m_softening);
    m_nbody->setDamping(compute.active_params.m_damping);

    if (compute.use_cpu) {
        reset_time_ = Clock::now();
    } else {
        checkCudaErrors(cudaEventCreate(&compute.start_event));
        checkCudaErrors(cudaEventCreate(&compute.stop_event));
        checkCudaErrors(cudaEventCreate(&compute.host_mem_sync_event));
    }

    if (!compute.benchmark && !compute.compare_to_cpu) {
        m_renderer = std::make_unique<ParticleRenderer>();
        _resetRenderer(compute.active_params.m_pointSize);
    }

    demo_reset_time_ = Clock::now();
    _reset(compute, config);
}

template <typename BodySystem>
template <typename>
NBodyDemo<BodySystem>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID) : tipsy_file_(std::move(tipsy_file)) {
    const auto use_pbo = !(compute.benchmark || compute.compare_to_cpu || compute.use_host_mem);
    m_nbody            = std::make_unique<BodySystem>(compute.num_bodies, numDevices, block_size, use_pbo, compute.use_host_mem, use_p2p, devID);

    const auto nb_bodies_4 = compute.num_bodies * 4;

    // allocate host memory
    m_hPos.resize(nb_bodies_4);
    m_hVel.resize(nb_bodies_4);
    m_hColor.resize(nb_bodies_4);

    m_nbody->setSoftening(compute.active_params.m_softening);
    m_nbody->setDamping(compute.active_params.m_damping);

    if (compute.use_cpu) {
        reset_time_ = Clock::now();
    } else {
        checkCudaErrors(cudaEventCreate(&compute.start_event));
        checkCudaErrors(cudaEventCreate(&compute.stop_event));
        checkCudaErrors(cudaEventCreate(&compute.host_mem_sync_event));
    }

    if (!compute.benchmark && !compute.compare_to_cpu) {
        m_renderer = std::make_unique<ParticleRenderer>();
        _resetRenderer(compute.active_params.m_pointSize);
    }

    demo_reset_time_ = Clock::now();
    _reset(compute, config);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::update_params(const NBodyParams& active_params) -> void {
    m_nbody->setSoftening(active_params.m_softening);
    m_nbody->setDamping(active_params.m_damping);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::update_simulation(float dt) -> void {
    m_nbody->update(dt);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_display(const ComputeConfig& compute, ParticleRenderer::DisplayMode display_mode) -> void {
    m_renderer->setSpriteSize(compute.active_params.m_pointSize);

    if (compute.use_host_mem) {
        // This event sync is required because we are rendering from the host memory that CUDA is writing.
        // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
        if (!compute.use_cpu) {
            cudaEventSynchronize(compute.host_mem_sync_event);
        }

        m_renderer->setPositions(m_nbody->get_position());
    } else {
        m_renderer->setPBO(m_nbody->getCurrentReadBuffer(), m_nbody->getNumBodies(), std::is_same_v<PrecisionType, double>);
    }

    // display particles
    m_renderer->display(display_mode);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::get_arrays(std::span<PrecisionType> pos, std::span<PrecisionType> vel) -> void {
    using std::ranges::copy;

    copy(m_nbody->get_position(), pos.begin());
    copy(m_nbody->get_velocity(), vel.begin());
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::set_arrays(std::span<const PrecisionType> pos, std::span<const PrecisionType> vel, const ComputeConfig& compute) -> void {
    m_nbody->set_position(pos);
    m_nbody->set_velocity(vel);

    if (!compute.benchmark && !compute.use_cpu && !compute.compare_to_cpu) {
        _resetRenderer(compute.active_params.m_pointSize);
    }
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_get_demo_time() -> float {
    return MilliSeconds{Clock::now() - demo_reset_time_}.count();
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_get_milliseconds_passed() -> float {
    const auto now          = Clock::now();
    const auto milliseconds = MilliSeconds{now - reset_time_}.count();

    reset_time_ = now;

    return milliseconds;
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_reset(ComputeConfig& compute, NBodyConfig config) -> void {
    if (tipsy_file_.empty()) {
        if constexpr (BodySystem::use_cpu) {
            randomise_bodies<BodySystem::Type>(config, m_nbody->get_position(), m_nbody->get_velocity(), m_hColor, compute.active_params.m_clusterScale, compute.active_params.m_velocityScale);
            if (!compute.benchmark && !compute.use_cpu && !compute.compare_to_cpu) {
                _resetRenderer(compute.active_params.m_pointSize);
            }
        } else {
            randomise_bodies<BodySystem::Type>(config, m_hPos, m_hVel, m_hColor, compute.active_params.m_clusterScale, compute.active_params.m_velocityScale);
            m_nbody->set_position(m_hPos);
            m_nbody->set_velocity(m_hVel);

            if (!compute.benchmark && !compute.use_cpu && !compute.compare_to_cpu) {
                _resetRenderer(compute.active_params.m_pointSize);
            }
        }

    } else {
        m_nbody->loadTipsyFile(tipsy_file_);
        compute.num_bodies = m_nbody->getNumBodies();
    }
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_resetRenderer(float point_size) -> void {
    const auto colour = std::is_same_v<PrecisionType, double> ? std::array{0.4f, 0.8f, 0.1f, 1.0f} : std::array{1.0f, 0.6f, 0.3f, 1.0f};

    m_renderer->setBaseColor(colour);
    m_renderer->setColors(m_hColor.data(), m_nbody->getNumBodies());
    m_renderer->setSpriteSize(point_size);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_selectDemo(ComputeConfig& compute) -> void {
    _reset(compute, NBodyConfig::NBODY_CONFIG_SHELL);

    demo_reset_time_ = Clock::now();
}

template NBodyDemo<BodySystemCPU<float>>;
template NBodyDemo<BodySystemCPU<double>>;
template NBodyDemo<BodySystemCUDA<float>>;
template NBodyDemo<BodySystemCUDA<double>>;

template NBodyDemo<BodySystemCPU<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
template NBodyDemo<BodySystemCPU<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
template NBodyDemo<BodySystemCUDA<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);
template NBodyDemo<BodySystemCUDA<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);