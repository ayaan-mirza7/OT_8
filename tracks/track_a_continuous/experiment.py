"""
Track A Benchmark Experiment Suite.
Runs rigorous comparative trials across all required continuous optimizers,
test landscapes, backprop neural network, and noisy/saddle environments.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_a_continuous.landscapes import ConvexQuadratic, SaddlePointLandscape, Rosenbrock, MultiMinimaRastrigin
from tracks.track_a_continuous.optimizers import (
    GradientDescent, Momentum, NesterovAcceleratedGradient, AdaGrad,
    RMSProp, Adam, NewtonRaphson, CurvatureApproximations
)
from tracks.track_a_continuous.neural_net import SmallMLP

def run_landscape_benchmark(landscape, landscape_name: str, x0: np.ndarray, n_iters: int = 150):
    results = {}
    optimizers = {
        'GD': GradientDescent(lr=0.05),
        'Momentum': Momentum(lr=0.05, beta=0.9),
        'NAG': NesterovAcceleratedGradient(lr=0.05, beta=0.9),
        'AdaGrad': AdaGrad(lr=0.2),
        'RMSProp': RMSProp(lr=0.05, beta=0.9),
        'Adam': Adam(lr=0.1, beta1=0.9, beta2=0.999),
        'Newton': NewtonRaphson(lr=1.0, damping=1e-2),
    }

    for name, opt in optimizers.items():
        x = x0.copy()
        history = [landscape.value(x)]
        start_t = time.perf_counter()

        for _ in range(n_iters):
            grad = landscape.gradient(x)
            if name == 'Newton':
                H = landscape.hessian(x)
                x = opt.step(x, grad, hessian=H)
            else:
                x = opt.step(x, grad)
            val = landscape.value(x)
            history.append(val)
            if np.isnan(val) or val > 1e7:
                break
        
        elapsed = time.perf_counter() - start_t
        results[name] = {
            'final_obj': float(history[-1]),
            'final_x': x.tolist(),
            'runtime_ms': float(elapsed * 1000.0),
            'history': [float(h) for h in history]
        }
    return results

def run_neural_net_benchmark(n_epochs: int = 80):
    # Synthetic binary classification dataset (XOR / non-linear circular boundary)
    rng = np.random.default_rng(42)
    N = 250
    X = rng.uniform(-2.0, 2.0, (N, 2))
    # Labels: inside radius 1.2 is 1, else 0
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.44).astype(np.float64).reshape(-1, 1)

    opt_constructors = {
        'GD': lambda: GradientDescent(lr=0.2),
        'Momentum': lambda: Momentum(lr=0.1, beta=0.9),
        'NAG': lambda: NesterovAcceleratedGradient(lr=0.1, beta=0.9),
        'AdaGrad': lambda: AdaGrad(lr=0.3),
        'RMSProp': lambda: RMSProp(lr=0.05),
        'Adam': lambda: Adam(lr=0.05)
    }

    mlp_results = {}
    for name, constructor in opt_constructors.items():
        mlp = SmallMLP(in_features=2, hidden_dim=8, out_features=1, seed=42)
        opt = constructor()
        loss_hist = []
        start_t = time.perf_counter()

        for _ in range(n_epochs):
            # Forward pass
            y_pred = mlp.forward(X)
            current_loss = mlp.loss(y_pred, y)
            loss_hist.append(current_loss)

            # Backpropagation (decoupled gradient computation)
            mlp.backward(y)

            # Optimizer update on flattened parameters
            params = mlp.get_flat_params()
            grads = mlp.get_flat_grads()
            new_params = opt.step(params, grads)
            mlp.set_flat_params(new_params)

        elapsed = time.perf_counter() - start_t
        final_preds = (mlp.forward(X) >= 0.5).astype(np.float64)
        accuracy = float(np.mean(final_preds == y))

        mlp_results[name] = {
            'final_loss': float(loss_hist[-1]),
            'accuracy': accuracy,
            'runtime_ms': float(elapsed * 1000.0),
            'history': loss_hist
        }
    return mlp_results

def run_noise_sensitivity_experiment():
    """Examines optimizer stability under noisy gradients on the Convex Quadratic objective."""
    landscape = ConvexQuadratic(dim=6, condition_number=10.0, seed=42)
    x0 = np.ones(6) * 2.5
    noise_levels = [0.0, 0.05, 0.2, 0.5]
    optimizers = {
        'GD': lambda: GradientDescent(lr=0.05),
        'Adam': lambda: Adam(lr=0.05),
        'Momentum': lambda: Momentum(lr=0.05, beta=0.9)
    }
    
    noise_results = []
    rng = np.random.default_rng(123)
    
    for sigma in noise_levels:
        for opt_name, factory in optimizers.items():
            opt = factory()
            x = x0.copy()
            for _ in range(100):
                grad = landscape.noisy_gradient(x, noise_std=sigma, rng=rng)
                x = opt.step(x, grad)
            final_val = landscape.value(x)
            noise_results.append({
                'noise_std': sigma,
                'optimizer': opt_name,
                'final_obj': final_val
            })
    return pd.DataFrame(noise_results)

def main():
    print("[Track A] Running Continuous Optimization Experiments...")
    os.makedirs("tracks/track_a_continuous/results", exist_ok=True)

    # 1. Convex Quadratic
    convex = ConvexQuadratic(dim=4, condition_number=15.0, seed=42)
    x0_convex = np.array([3.0, -2.5, 4.0, -1.5])
    convex_res = run_landscape_benchmark(convex, "Convex Quadratic", x0_convex, n_iters=120)

    # 2. Saddle Point Landscape
    saddle = SaddlePointLandscape()
    # Start near saddle point (0.01, 0.05) to observe escape dynamics
    x0_saddle = np.array([0.05, 0.02])
    saddle_res = run_landscape_benchmark(saddle, "Saddle Point", x0_saddle, n_iters=100)

    # 3. Rosenbrock
    rosen = Rosenbrock()
    x0_rosen = np.array([-1.2, 1.0])
    rosen_res = run_landscape_benchmark(rosen, "Rosenbrock", x0_rosen, n_iters=180)

    # 4. Neural Network with Backprop
    mlp_res = run_neural_net_benchmark(n_epochs=100)

    # 5. Noise Sensitivity
    noise_df = run_noise_sensitivity_experiment()
    noise_df.to_csv("tracks/track_a_continuous/results/noise_sensitivity.csv", index=False)

    # Generate Summary Table
    rows = []
    for opt in convex_res.keys():
        rows.append({
            'Optimizer': opt,
            'Convex_Final_Obj': convex_res[opt]['final_obj'],
            'Saddle_Final_Obj': saddle_res[opt]['final_obj'],
            'Rosenbrock_Final_Obj': rosen_res[opt]['final_obj'],
            'Convex_Runtime_ms': convex_res[opt]['runtime_ms']
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("tracks/track_a_continuous/results/benchmark_summary.csv", index=False)

    mlp_df = pd.DataFrame([
        {'Optimizer': k, 'Final_Loss': v['final_loss'], 'Accuracy': v['accuracy'], 'Runtime_ms': v['runtime_ms']}
        for k, v in mlp_res.items()
    ])
    mlp_df.to_csv("tracks/track_a_continuous/results/mlp_benchmark.csv", index=False)

    # Generate Publication Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Track A: Continuous Optimization Convergence Dynamics\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Convex Quadratic
    ax = axes[0, 0]
    for opt, data in convex_res.items():
        ax.plot(data['history'][:80], label=opt, lw=1.8)
    ax.set_yscale('log')
    ax.set_title("Convex Quadratic Convergence (log scale)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Saddle Point Landscape
    ax = axes[0, 1]
    for opt, data in saddle_res.items():
        ax.plot(data['history'][:80], label=opt, lw=1.8)
    ax.set_title("Saddle Point Escape Dynamics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Rosenbrock
    ax = axes[1, 0]
    for opt, data in rosen_res.items():
        ax.plot(data['history'][:120], label=opt, lw=1.8)
    ax.set_yscale('log')
    ax.set_title("Rosenbrock Valley Optimization (log scale)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: MLP Training Loss
    ax = axes[1, 1]
    for opt, data in mlp_res.items():
        ax.plot(data['history'], label=f"{opt} (Acc: {data['accuracy']*100:.1f}%)", lw=1.8)
    ax.set_title("MLP Backprop Loss Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "tracks/track_a_continuous/results/track_a_convergence_plots.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[Track A] Completed successfully. Artifacts saved in tracks/track_a_continuous/results/")
    return summary_df

if __name__ == '__main__':
    main()
