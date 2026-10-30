# Prompt Log: Track A (Continuous Optimization)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track A — Continuous Optimization & Curvature

## 1. Initial Prompt
> "Implement all continuous optimizers required by Chapter 6 in Python from scratch: Gradient Descent, Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, Adam, Newton-Raphson with Levenberg-Marquardt damping, and Gauss-Newton for least squares. Also include curvature approximations (Fisher diagonal and Pearlmutter Hessian-Vector Products). Build test landscapes: strongly convex quadratic, a saddle-point landscape with escaping directions, Rosenbrock valley, and a small 2-layer MLP demonstrating that backpropagation is strictly decoupled from the optimizer."

## 2. AI Response Summary
The AI assistant drafted `landscapes.py`, `optimizers.py`, and `neural_net.py`. It defined class structures for each optimizer with standard gradient descent formulas and an MLP backprop pass computing analytical gradients using the multivariable chain rule.

## 3. Revised Prompt (Iterative Refinement)
> "Ensure that the Newton-Raphson optimizer incorporates explicit Levenberg-Marquardt diagonal damping ($\mu I$) so that it does not diverge on non-convex saddle regions where the Hessian has negative eigenvalues. Also add Pearlmutter finite-difference Hessian-Vector Products to the landscape class so second-order steps can be computed without forming the full $O(N^2)$ Hessian."

## 4. Resulting Code Changes
- Added `hessian_vector_product` in `landscapes.py`.
- Added LM adaptive damping $\mu = 10^{-2}$ to `NewtonRaphson.step` in `optimizers.py`.
- Verified strict parameter decoupling: `mlp.backward(y)` updates only `mlp.grads` dict without mutating parameters.

## 5. Verification & Test Evidence
- Verified via `test_track_a_convex_quadratic_gradient` and `test_track_a_mlp_backpropagation_decoupling`.
- Benchmark outputs saved to `tracks/track_a_continuous/results/benchmark_summary.csv` and convergence plot `track_a_convergence_plots.png`.
