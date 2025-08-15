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

    m_nbody->setSoftening(compute.active_params.m_softening);
    m_nbody->setDamping(compute.active_params.m_damping);

    _reset(compute, config, {});
    compute.num_bodies = m_nbody->getNumBodies();
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

    m_nbody->setSoftening(compute.active_params.m_softening);
    m_nbody->setDamping(compute.active_params.m_damping);

    _reset(compute, config, {});
    compute.num_bodies = m_nbody->getNumBodies();
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::update_params(const NBodyParams& active_params) -> void {
    m_nbody->setSoftening(active_params.m_softening);
    m_nbody->setDamping(active_params.m_damping);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::update_simulation(float dt) -> void {
    m_nbody->update(dt);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::get_arrays(std::span<PrecisionType> pos, std::span<PrecisionType> vel) -> void {
    using std::ranges::copy;

    copy(m_nbody->get_position(), pos.begin());
    copy(m_nbody->get_velocity(), vel.begin());
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::set_arrays(std::span<const PrecisionType> pos, std::span<const PrecisionType> vel) -> void {
    m_nbody->set_position(pos);
    m_nbody->set_velocity(vel);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_reset(const ComputeConfig& compute, NBodyConfig config, std::span<float> colour) -> void {
    if (tipsy_file_.empty()) {
        if constexpr (BodySystem::use_cpu) {
            randomise_bodies<BodySystem::Type>(config, m_nbody->get_position(), m_nbody->get_velocity(), colour, compute.active_params.m_clusterScale, compute.active_params.m_velocityScale);
        } else {
            randomise_bodies<BodySystem::Type>(config, m_hPos, m_hVel, colour, compute.active_params.m_clusterScale, compute.active_params.m_velocityScale);
            m_nbody->set_position(m_hPos);
            m_nbody->set_velocity(m_hVel);
        }

    } else {
        m_nbody->loadTipsyFile(tipsy_file_);
    }
}

template NBodyDemo<BodySystemCPU<float>>;
template NBodyDemo<BodySystemCPU<double>>;
template NBodyDemo<BodySystemCUDA<float>>;
template NBodyDemo<BodySystemCUDA<double>>;

template NBodyDemo<BodySystemCPU<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
template NBodyDemo<BodySystemCPU<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config);
template NBodyDemo<BodySystemCUDA<float>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);
template NBodyDemo<BodySystemCUDA<double>>::NBodyDemo(std::filesystem::path tipsy_file, ComputeConfig& compute, NBodyConfig config, int numDevices, int block_size, bool use_p2p, int devID);