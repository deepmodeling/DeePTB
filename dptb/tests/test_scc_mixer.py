import numpy as np

from dptb.nn.dftb.scc_mixer import DIISMixer


def test_diis_mixer_reset_restores_initial_alpha():
    mixer = DIISMixer(n_generations=2, alpha=0.4)
    mixer.reset(n_elem=2)

    mixer.alpha = 0.9
    mixer.reset(n_elem=2)

    assert mixer.alpha == 0.4


def test_diis_mixer_adaptive_alpha_is_clamped():
    mixer = DIISMixer(n_generations=2, alpha=0.9)
    mixer.reset(n_elem=2)
    mixer.i_prev_vector = 1
    mixer.prev_q_input[:, 0] = np.array([0.2, 0.8])
    mixer.prev_q_diff[:, 0] = np.array([1.0, 0.0])
    mixer.delta_r = np.array([1.0, 0.0])
    q_inp = np.array([0.2, 0.8])
    q_diff = np.array([1.0, 0.0])

    mixer._apply_diis(q_inp, q_diff)

    assert mixer.alpha == 1.0
