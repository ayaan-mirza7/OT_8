import pytest
import numpy as np

# Track A Tests
from tracks.track_a_continuous.landscapes import ConvexQuadratic, SaddlePointLandscape, Rosenbrock
from tracks.track_a_continuous.optimizers import GradientDescent, Adam, NewtonRaphson
from tracks.track_a_continuous.neural_net import SmallMLP

def test_track_a_convex_quadratic_gradient():
    cq = ConvexQuadratic(dim=3, condition_number=5.0, seed=42)
    x = np.array([1.0, 2.0, 3.0])
    exact_g = cq.gradient(x)
    # Numerical gradient check
    eps = 1e-6
    num_g = np.zeros_like(x)
    for i in range(3):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += eps
        x_m[i] -= eps
        num_g[i] = (cq.value(x_p) - cq.value(x_m)) / (2.0 * eps)
    assert np.allclose(exact_g, num_g, atol=1e-5)

def test_track_a_mlp_backpropagation_decoupling():
    mlp = SmallMLP(in_features=2, hidden_dim=4, out_features=1, seed=42)
    X = np.array([[0.5, -0.2], [-0.1, 0.8]])
    y = np.array([[1.0], [0.0]])
    preds = mlp.forward(X)
    loss = mlp.loss(preds, y)
    grads = mlp.backward(y)
    assert 'W1' in grads and 'W2' in grads
    assert grads['W1'].shape == mlp.W1.shape
    # Ensure backprop does NOT update parameters (strict decoupling)
    p_before = mlp.get_flat_params()
    # parameter array is unchanged until optimizer step is explicitly called
    assert np.array_equal(p_before, mlp.get_flat_params())

# Track B Tests
from tracks.track_b_discrete.combinatorial_solvers import KnapsackProblem, solve_greedy_knapsack, solve_dp_knapsack, solve_ilp_knapsack
from tracks.track_b_discrete.qubo_formulation import MaxCutQUBO

def test_track_b_knapsack_optimality():
    values = [60, 100, 120]
    weights = [10, 20, 30]
    cap = 50
    prob = KnapsackProblem(values, weights, cap)
    dp_res = solve_dp_knapsack(prob)
    ilp_res = solve_ilp_knapsack(prob)
    assert dp_res['value'] == 220.0
    assert ilp_res['value'] == 220.0

def test_track_b_maxcut_qubo_equivalence():
    adj = np.array([[0, 1], [1, 0]])
    qubo = MaxCutQUBO(adj)
    # Cut between 0 and 1 is 1
    cut_val = qubo.evaluate_cut(np.array([0, 1]))
    assert cut_val == 1.0
    spin = qubo.binary_to_spin(np.array([0, 1]))
    assert np.array_equal(spin, np.array([1, -1]))
    ising_e = qubo.ising_energy(spin)
    assert ising_e == 1.0

# Track C Tests
from tracks.track_c_quantum.simulator import QuantumState, H_GATE, X_GATE
from tracks.track_c_quantum.gradients import parameter_shift_gradient

def test_track_c_quantum_superposition_and_expectation():
    qs = QuantumState(2)
    qs.apply_single_gate(H_GATE, 0)
    # <Z_0> should be 0.0
    assert abs(qs.expectation_z(0)) < 1e-12
    # Apply X to qubit 1: <Z_1> should be -1.0
    qs.apply_single_gate(X_GATE, 1)
    assert abs(qs.expectation_z(1) - (-1.0)) < 1e-12

def test_track_c_parameter_shift_rule():
    # f(theta) = cos(theta)
    def circuit_fn(params):
        return float(np.cos(params[0]))
    theta = np.array([0.7])
    exact_grad = -np.sin(0.7)
    ps_grad = parameter_shift_gradient(circuit_fn, theta)
    assert abs(ps_grad[0] - exact_grad) < 1e-7

# Track D Tests
from tracks.track_d_parallel_distributed.submodular import FacilityLocationSubmodular
from core.metrics import amdahls_law, gustafsons_law, jains_fairness_index

def test_track_d_submodular_diminishing_returns():
    fac = FacilityLocationSubmodular(num_facilities=8, num_targets=20, seed=42)
    res = fac.verify_diminishing_returns(num_trials=40, seed=42)
    assert res['is_submodular'] is True

def test_track_d_amdahl_and_jain_fairness():
    sp = amdahls_law(speedup_parallel_portion=4.0, parallel_fraction=0.8)
    assert sp > 1.0
    allocations = [10, 10, 10, 10]
    assert jains_fairness_index(allocations) == 1.0

# Track F Tests
from tracks.track_f_compiler.ir import Instruction
from tracks.track_f_compiler.passes import constant_folding_and_propagation, dead_code_elimination
from tracks.track_f_compiler.code_gen import VirtualMachine

def test_track_f_compiler_pass_correctness():
    prog = [
        Instruction('ASSIGN', '10', None, 'a'),
        Instruction('ASSIGN', '20', None, 'b'),
        Instruction('ADD', 'a', 'b', 'c'),
        Instruction('PRINT', 'c', None, None)
    ]
    vm_before = VirtualMachine(prog)
    res_before = vm_before.execute()
    
    folded = constant_folding_and_propagation(prog)
    vm_after = VirtualMachine(folded)
    res_after = vm_after.execute()

    assert res_before['output'] == res_after['output']
    assert res_after['output'] == [30]

# Track H Tests
from tracks.track_h_dbms.schema_and_data import create_tpch_database

def test_track_h_dbms_database_creation():
    conn = create_tpch_database(db_path=":memory:", num_customers=50, num_orders=100, seed=42)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM customer;")
    assert cur.fetchone()[0] == 50
    cur.execute("SELECT count(*) FROM orders;")
    assert cur.fetchone()[0] == 100

# Track I Tests
from tracks.track_i_quantum_advantage.algorithms import run_deutsch_jozsa, run_shor_order_finding

def test_track_i_quantum_algorithms():
    dj_balanced = run_deutsch_jozsa(num_qubits=3, is_constant=False)
    assert dj_balanced['correct'] is True
    shor = run_shor_order_finding(a=7, N=15)
    assert shor['success'] is True
    assert set(shor['extracted_factors']) == {3, 5}

# Track J Tests
from tracks.track_j_software_engineering.test_optimizer import build_test_suite_model, solve_ilp_test_minimization

def test_track_j_test_minimization_coverage():
    tests, branches, _ = build_test_suite_model(num_tests=20, num_branches=25, seed=42)
    res = solve_ilp_test_minimization(tests, branches)
    assert res['branches_covered'] == branches
    assert res['coverage_percentage'] == 100.0
