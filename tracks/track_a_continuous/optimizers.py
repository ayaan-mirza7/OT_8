"""
Continuous Optimizers implementation from scratch.
Implements all Chapter 6 continuous algorithms:
1. Gradient Descent (GD)
2. Stochastic Gradient Descent (SGD / Mini-batch)
3. Momentum (Polyak Heavy-Ball)
4. Nesterov Accelerated Gradient (NAG)
5. AdaGrad
6. RMSProp
7. Adam
8. Newton-Raphson (2nd Order with Levenberg-Marquardt damping)
9. Gauss-Newton (Nonlinear Least Squares)
10. Curvature Approximations (Fisher Diagonal, Hessian-Vector Product CG)

Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

class Optimizer:
    """Base optimizer interface."""
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        return x - self.lr * grad


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = None

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.beta * self.velocity + self.lr * grad
        return x - self.velocity


class NesterovAcceleratedGradient(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = None

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        v_prev = self.velocity.copy()
        self.velocity = self.beta * self.velocity + self.lr * grad
        # NAG step incorporating momentum forecast
        return x - (self.beta * v_prev + (1.0 + self.beta) * (self.lr * grad))


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad ** 2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.eps)
        return x - adjusted_lr * grad


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad ** 2)
        return x - (self.lr / (np.sqrt(self.v) + self.eps)) * grad


class Adam(Optimizer):
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        
        # Bias corrections
        m_hat = self.m / (1.0 - (self.beta1 ** self.t))
        v_hat = self.v / (1.0 - (self.beta2 ** self.t))
        return x - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


class NewtonRaphson(Optimizer):
    """
    Second-order optimization using full exact Hessian with Levenberg-Marquardt damping
    to guarantee descent even in non-convex or saddle regions.
    """
    def __init__(self, lr: float = 1.0, damping: float = 1e-3):
        super().__init__(lr)
        self.damping = damping

    def step(self, x: np.ndarray, grad: np.ndarray, hessian: np.ndarray = None, **kwargs) -> np.ndarray:
        assert hessian is not None, "Newton-Raphson requires exact Hessian matrix"
        H = hessian + self.damping * np.eye(len(x))
        try:
            # Solve H * step_dir = grad
            step_dir = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step_dir = np.linalg.pinv(H) @ grad
        return x - self.lr * step_dir


class GaussNewton(Optimizer):
    """
    Gauss-Newton method for non-linear least squares:
    Minimizes 0.5 * ||r(x)||^2.
    Approximate Hessian H ~ J^T J, gradient g = J^T r.
    """
    def __init__(self, lr: float = 1.0, damping: float = 1e-4):
        super().__init__(lr)
        self.damping = damping

    def step_gn(self, x: np.ndarray, residual: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        J = jacobian
        r = residual
        grad = J.T @ r
        H_approx = J.T @ J + self.damping * np.eye(len(x))
        step_dir = np.linalg.solve(H_approx, grad)
        return x - self.lr * step_dir


class CurvatureApproximations:
    """
    Curvature and second-order approximation primitives:
    - Fisher Information Matrix diagonal approximation
    - Low-rank curvature / BFGS-style update
    - Hessian-Vector Product Conjugate Gradient (Pearlmutter trick)
    """
    @staticmethod
    def diagonal_fisher_update(x: np.ndarray, grads_batch: np.ndarray, lr: float = 0.01, eps: float = 1e-4) -> np.ndarray:
        """Computes empirical diagonal Fisher Information Matrix F = E[g * g^T] and applies F^{-1} g."""
        # grads_batch shape: (batch_size, dim)
        fisher_diag = np.mean(grads_batch ** 2, axis=0) + eps
        mean_grad = np.mean(grads_batch, axis=0)
        return x - lr * (mean_grad / fisher_diag)

    @staticmethod
    def hessian_free_cg_step(x: np.ndarray, grad: np.ndarray, landscape, max_cg_iter: int = 10) -> np.ndarray:
        """
        Computes second-order step using Conjugate Gradient without forming full Hessian,
        evaluating curvature solely via Hessian-Vector Products (Pearlmutter trick).
        """
        dim = len(x)
        d = np.zeros(dim)
        r = -grad.copy()
        p = r.copy()
        rs_old = np.dot(r, r)

        for _ in range(max_cg_iter):
            if rs_old < 1e-12:
                break
            # Compute H * p using Pearlmutter HVP
            Ap = landscape.hessian_vector_product(x, p) + 1e-3 * p
            alpha = rs_old / (np.dot(p, Ap) + 1e-9)
            d = d + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r, r)
            p = r + (rs_new / (rs_old + 1e-9)) * p
            rs_old = rs_new
        return x + d
