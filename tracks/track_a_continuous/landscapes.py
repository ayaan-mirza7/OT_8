"""
Optimization Landscapes for Track A: Continuous Optimization.
Includes Strongly Convex, Convex, and Non-Convex (with saddle points and local minima) functions.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

class OptimizationLandscape:
    """Base class for test objective functions."""
    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Finite-difference / Pearlmutter approximation of H * v."""
        grad_plus = self.gradient(x + eps * v)
        grad_minus = self.gradient(x - eps * v)
        return (grad_plus - grad_minus) / (2.0 * eps)

    def noisy_gradient(self, x: np.ndarray, noise_std: float = 0.1, rng: np.random.Generator = None) -> np.ndarray:
        """Simulates stochastic minibatch gradient noise."""
        if rng is None:
            rng = np.random.default_rng()
        grad = self.gradient(x)
        noise = rng.normal(0.0, noise_std, size=grad.shape)
        return grad + noise


class ConvexQuadratic(OptimizationLandscape):
    """
    Strongly Convex Quadratic: f(x) = 0.5 * x^T A x - b^T x
    Allows precise control over condition number kappa = lambda_max / lambda_min.
    """
    def __init__(self, dim: int = 10, condition_number: float = 20.0, seed: int = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Generate orthogonal matrix Q
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        eigenvalues = np.linspace(1.0, condition_number, dim)
        self.A = Q @ np.diag(eigenvalues) @ Q.T
        # Symmetrize
        self.A = 0.5 * (self.A + self.A.T)
        self.x_opt = rng.standard_normal(dim)
        self.b = self.A @ self.x_opt

    def value(self, x: np.ndarray) -> float:
        diff = x - self.x_opt
        return float(0.5 * diff.T @ self.A @ diff)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ (x - self.x_opt)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A.copy()

    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return self.A @ v


class SaddlePointLandscape(OptimizationLandscape):
    """
    Non-Convex landscape featuring an escape-critical saddle point at origin:
    f(x1, x2) = x1^2 - x2^2 + 0.1 * x2^4
    Hessian at (0, 0) is diag([2, -2]), having one positive and one negative eigenvalue.
    Local minima are at (0, +/- sqrt(5)).
    """
    def value(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return float(x1**2 - x2**2 + 0.1 * (x2**4))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        g1 = 2.0 * x1
        g2 = -2.0 * x2 + 0.4 * (x2**3)
        return np.array([g1, g2], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        h11 = 2.0
        h22 = -2.0 + 1.2 * (x2**2)
        return np.array([[h11, 0.0], [0.0, h22]], dtype=np.float64)


class Rosenbrock(OptimizationLandscape):
    """
    Non-Convex classic Rosenbrock banana valley:
    f(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2
    Global minimum at (1, 1) with value 0. High curvature valley tests adaptive methods.
    """
    def value(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return float(100.0 * (x2 - x1**2)**2 + (1.0 - x1)**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        g1 = -400.0 * x1 * (x2 - x1**2) - 2.0 * (1.0 - x1)
        g2 = 200.0 * (x2 - x1**2)
        return np.array([g1, g2], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        h11 = -400.0 * (x2 - 3.0 * x1**2) + 2.0
        h12 = -400.0 * x1
        h22 = 200.0
        return np.array([[h11, h12], [h12, h22]], dtype=np.float64)


class MultiMinimaRastrigin(OptimizationLandscape):
    """
    Highly non-convex multimodal landscape with numerous local minima:
    f(x) = 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    Global minimum at x = 0 with value 0.
    """
    def __init__(self, dim: int = 4):
        self.dim = dim

    def value(self, x: np.ndarray) -> float:
        return float(10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        diag = 2.0 + 40.0 * (np.pi**2) * np.cos(2.0 * np.pi * x)
        return np.diag(diag)
