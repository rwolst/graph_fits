import numpy as np

from QCV import (sinkhorn_balancing, objective_f, compute_gradient,
                 optimal_ds, find_search_direction, optimal_step_size)


def ensure_ds(X):
    """Ensure matrix X is doubly stochastic."""
    assert np.all((0 <= X) & (X <= 1))
    assert np.all((np.sum(X, 0) - 1)**2 < 1e-3)
    assert np.all((np.sum(X, 1) - 1)**2 < 1e-3)


def test_sinkhron_balancing():
    """Make sure Sinkhorn balancing is able to make doubly stochastic
    matrices."""
    k = 4

    X = np.random.rand(k, k)

    X = sinkhorn_balancing(X, 1000, 1e-3)

    # Make sure doubly stochastic.
    ensure_ds(X)


def test_grad():
    """Test the gradient is close to a numeric approximation."""
    k = 4

    A = np.random.rand(k, k)
    B = np.random.rand(k, k)
    Q = np.random.rand(k, k)  # Doesn't need to be DS for this test.

    grad = compute_gradient(A, B, Q)
    grad_num = np.empty([k,k])

    # Now get numeric gradient for each Q value.
    delta = 0.000001
    base = objective_f(A, B, Q)
    for i in range(k):
        for j in range(k):
            Q_new = Q.copy()
            Q_new[i, j] += delta

            prop = objective_f(A, B, Q_new)
            grad_num[i, j] = (prop - base)/delta

    assert np.all((grad - grad_num)**2 < 1e-3)


def test_find_search_direction():
    """Make sure the found direction W, maximises
        tr(grad @ W)
    for all doubly stochastic/permutation matrices.
    """
    k = 4
    grad = np.random.rand(k, k)

    W = find_search_direction(grad, k)

    # Make sure W is a permutation matrix. Use a trick that x*(1-x) has roots
    # at 0 and 1.
    assert np.all(W * (1-W) == 0)
    assert np.all(np.sum(W, 0) == 1)
    assert np.all(np.sum(W, 1) == 1)

    # For different doubly stochastic make sure they are not less than
    #     tr(grad.T @ W).
    base = np.trace(grad.T @ W)
    n_samples = 20
    for sample in range(n_samples):
        X = np.random.rand(k, k)
        X = sinkhorn_balancing(X, 1000, 1e-3)

        prop = np.trace(grad.T @ X)

        assert base < prop


def test_optimal_step_size():
    """Make sure that alpha is the best value for the objective function."""
    k = 4

    A = np.random.rand(k, k)
    B = np.random.rand(k, k)
    Q = np.random.rand(k, k)
    W = np.random.rand(k, k)

    alpha = optimal_step_size(A, B, Q, W)

    base = objective_f(A, B, (1-alpha)*Q + alpha*W)

    n_samples = 20
    for sample in range(n_samples):
        alpha = np.random.rand()

        prop = objective_f(A, B, (1-alpha)*Q + alpha*W)
        assert base < prop


def test_optimal_ds():
    """Test that
        1) Returned matrix is in fact doubly stochastic.
        2) The optimal doubly stochastic matrix Q that is returned is better
           than a number of randomly chosen doubly stochastic matrices.
        3) Different inital starting points Q0 lead to the same Q being
           returned.
           """
    k = 40

    A = np.random.rand(k, k)
    B = np.random.rand(k, k)

    Q = optimal_ds(A, B, 1000, 1e-4, verbose=True)

    # Make sure doubly stochastic.
    ensure_ds(Q)

    # Random matrices.
    base = objective_f(A, B, Q)
    n_samples = 5
    for sample in range(n_samples):
        Q_prop = sinkhorn_balancing(np.random.rand(k, k), 10000, 1e-4)
        prop = objective_f(A, B, Q_prop)

        assert base < prop

    # Test for different starting points we get same optimal Q.
    n_samples = 5
    for sample in range(n_samples):
        Q0 = sinkhorn_balancing(np.random.rand(k, k), 1000, 1e-3)
        Q_prop = optimal_ds(A, B, 10000, 1e-4, Q0=Q0)

        assert np.all((Q - Q_prop)**2 < 1e-3)
