"""
Quantum Gradient Calculation Methods:
1. Exact Parameter-Shift Rule (d <H> / d theta = 0.5 * [<H>(theta + pi/2) - <H>(theta - pi/2)])
2. Simultaneous Perturbation Stochastic Approximation (SPSA)
3. Quantum Natural Gradient (QNG using Fubini-Study Quantum Fisher Information metric)
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

def parameter_shift_gradient(circuit_fn, params: np.ndarray, shift: float = np.pi / 2.0) -> np.ndarray:
    """Computes exact analytical gradients using the two-point parameter-shift rule."""
    grads = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        val_plus = circuit_fn(params_plus)
        val_minus = circuit_fn(params_minus)
        grads[i] = 0.5 * (val_plus - val_minus)
    return grads

def spsa_gradient_estimate(circuit_fn, params: np.ndarray, c_k: float = 0.1, rng: np.random.Generator = None) -> np.ndarray:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA).
    Requires exactly TWO function evaluations regardless of parameter dimension.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Bernoulli +/- 1 perturbation vector
    delta = rng.choice([-1.0, 1.0], size=len(params))
    
    val_plus = circuit_fn(params + c_k * delta)
    val_minus = circuit_fn(params - c_k * delta)
    
    # g_i = (y+ - y-) / (2 * c_k * delta_i)
    grad_estimate = (val_plus - val_minus) / (2.0 * c_k * delta)
    return grad_estimate

def quantum_natural_gradient_step(
    circuit_state_fn,
    hamiltonian_fn,
    params: np.ndarray,
    lr: float = 0.05,
    damping: float = 1e-3,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Performs a Quantum Natural Gradient (QNG) step:
    theta_{t+1} = theta_t - lr * (g(theta) + damping * I)^{-1} * grad(theta)
    where g(theta) is the Fubini-Study metric tensor approximation.
    """
    dim = len(params)
    psi0 = circuit_state_fn(params)
    grad = parameter_shift_gradient(hamiltonian_fn, params)

    # Compute Fubini-Study metric tensor g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
    d_psi = []
    for i in range(dim):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        d_psi_i = (circuit_state_fn(p_plus) - circuit_state_fn(p_minus)) / (2.0 * eps)
        d_psi.append(d_psi_i)

    g_matrix = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            overlap_ij = np.vdot(d_psi[i], d_psi[j])
            overlap_i0 = np.vdot(d_psi[i], psi0)
            overlap_0j = np.vdot(psi0, d_psi[j])
            qfi = np.real(overlap_ij - overlap_i0 * overlap_0j)
            g_matrix[i, j] = qfi

    g_reg = g_matrix + damping * np.eye(dim)
    try:
        natural_step = np.linalg.solve(g_reg, grad)
    except np.linalg.LinAlgError:
        natural_step = np.linalg.pinv(g_reg) @ grad

    return params - lr * natural_step
