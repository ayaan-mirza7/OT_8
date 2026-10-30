# Prompt Log: Track C (Quantum Optimization)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track C — Quantum Optimization (QAOA, VQE, Gradients)

## 1. Initial Prompt
> "Build a pure Python + NumPy statevector quantum simulator supporting single-qubit gates, CNOT, and Rzz interaction gates. Implement QAOA to solve the exact Max-Cut graph instance saved from Track B. Implement the two-point Parameter-Shift rule, SPSA stochastic gradient estimation, and Quantum Natural Gradient (QNG) using the Fubini-Study metric tensor. Implement VQE for ground-state energy calculation, and study the Barren Plateau phenomenon by measuring gradient variance decay as qubit count grows."

## 2. AI Response Summary
The assistant created `simulator.py`, `qaoa.py`, `vqe.py`, `gradients.py`, and `barren_plateau.py`. It formulated QAOA with alternating cost unitaries $e^{-i \gamma H_C}$ and mixer unitaries $e^{-i \beta H_M}$, and implemented exact analytical gradients using the parameter shift rule $\frac{1}{2}[\langle H \rangle(\theta + \pi/2) - \langle H \rangle(\theta - \pi/2)]$.

## 3. Revised Prompt
> "Ensure that the simulator includes a depolarizing noise model to study the impact of gate errors on QAOA cut expectation, and add multinomial projective measurement sampling to demonstrate finite-shot sampling variance."

## 4. Resulting Code Changes
- Implemented `apply_depolarizing_channel` and `sample_measurements` in `QuantumState`.
- Successfully linked QAOA to Track B's JSON instance; verified that QAOA achieves a 0.94 approximation ratio relative to the classical exact maximum cut.
- Verified exponential decay of gradient variance $Var[\partial_\theta \langle H \rangle]$ across $N \in [2, 4, 6, 8]$ qubits, demonstrating the Barren Plateau theorem.
