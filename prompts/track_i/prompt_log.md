# Prompt Log: Track I (Quantum Algorithms & QML Advantage)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track I — Quantum Algorithms and QML Advantage

## 1. Initial Prompt
> "Implement the full suite of quantum algorithms taught in class from scratch: Deutsch-Jozsa, Bernstein-Vazirani, Grover's Amplitude Amplification, Quantum Fourier Transform (QFT), Quantum Phase Estimation (QPE), Shor's Order-Finding algorithm (factoring 15 into 3 x 5), and a Quantum Annealing simulator. Train a Variational Quantum Classifier (VQC) with angle embedding and compare it against an equivalent Classical MLP on accuracy, training time, parameter count, and circuit evaluations. Compare Grover search against classical linear search in queries vs wall-clock time, explaining the honest negative result regarding classical simulation overhead."

## 2. AI Response & Observed Error
The assistant drafted `algorithms.py`, `qml_classifier.py`, `complexity_comparison.py`, and `experiment.py`. During the classical classifier baseline execution, an array shape error occurred: `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 (size 1 is different from 40)` in `neural_net.py` because `y` was passed as 1D `(40,)` instead of 2D column `(40, 1)`.

## 3. Revised Prompt (Correction)
> "In `run_classical_classifier_baseline`, reshape target vector `y` into a 2D column `y_col = y.reshape(-1, 1)` before calling `mlp.backward(y_col)` and `mlp.loss(..., y_col)`."

## 4. Resulting Code Changes
- Fixed array dimensionality in `qml_classifier.py`.
- Verified all 7 quantum algorithms (100% success on Shor order-finding, QFT unitarity, Deutsch-Jozsa, and Bernstein-Vazirani secret extraction).
- Quantified Grover algorithmic query advantage ($O(\sqrt{N})$ vs $O(N)$) alongside the honest negative result: simulated quantum statevector evolution on classical hardware suffers an exponential wall-clock time overhead of up to $225\times$.
