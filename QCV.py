"""Module for implementing convex optimisations over doubly stochastic
matrices."""

import numpy as np
import scipy as sp
import scipy.optimize
from contexttimer import Timer
from lapsolver import solve_dense
from munkres import Munkres


def sinkhorn_balancing(X, max_iter, tol):
    """Balances the matrix X into a doubly stocahstic matrix using Sinkhorn
    balancing."""
    conv = False
    for i in range(max_iter):
        # Normalise by columns.
        X = X/np.sum(X, 0)[None, :]

        # Normalise by rows.
        X = X/np.sum(X, 1)[:, None]

        # Convergence check.
        col_conv = np.all((np.sum(X, 0) - 1)**2 < tol)
        row_conv = np.all((np.sum(X, 1) - 1)**2 < tol)
        if col_conv and row_conv:
            conv = True
            break

    if not conv:
        print("Warning, hit maximum iteration in sinkhorn_balancing.")

    return X


def objective_f(A, B, Q):
    """Calculates the objective function."""
    obj_f = (np.linalg.norm(Q @ A - B @ Q, 'fro'))**2

    return obj_f


def compute_gradient(A, B, Q):
    """Compute gradient of objective w.r.t Q."""
    grad = 2 * Q @ (A @ A.T) + \
           2 * (B.T @ B) @ Q - \
           2 * (B.T @ Q) @ A - \
           2 * (B @ Q) @ A.T

    return grad


def find_search_direction(grad, k):
    """Find the search direction by finding the direction in the doubly
    stochastic matrices where grad is minimised."""
    # Minimise the negative of the grad.
    #row_ind, col_ind = sp.optimize.linear_sum_assignment(grad)
    row_ind, col_ind = solve_dense(grad)
    #m = Munkres()
    #indices = np.array(m.compute(grad))
    #row_ind = indices[:,0]
    #col_ind = indices[:,1]
    W = np.zeros([k,k])
    W[row_ind, col_ind] = 1

    return W


def optimal_step_size(A, B, Q, W):
    """Calculates optimal step size alpha in the Frank Wolfe algorithm to
    minimise
        f(Q + alpha W)
    where
        f(Q) = || Q A - B Q ||_{Fro}.

    Note:
        This can be sped up by calculating commonly occuring pairs of matrices
        e.g. A.T @ Q.T and then doing an elementwise multiply and sum, which is
        equivalent to tr[X @ Y] (where X and Y are example matrices).
    """
    Z = W - Q
    A_T_Q_T = A.T @ Q.T
    A_T_Z_T = A.T @ Z.T
    Z_T_B_T = Z.T @ B.T
    Q_T_B_T = Q.T @ B.T

    #import pdb; pdb.set_trace()
    numerator = np.sum(
                    + A_T_Q_T * Z_T_B_T \
                    + A_T_Z_T * Q_T_B_T \
                    - A_T_Q_T * A_T_Z_T \
                    - Q_T_B_T * Z_T_B_T)

    denominator = np.sum(
                      + A_T_Z_T * A_T_Z_T \
                      + Z_T_B_T * Z_T_B_T \
                      - 2*A_T_Z_T * Z_T_B_T)

    #numerator = np.trace(
    #                + (A.T @ Q.T @ B @ Z) \
    #                + (A.T @ Z.T @ B @ Q) \
    #                - (A.T @ Q.T @ Z @ A) \
    #                - (Q.T @ B.T @ B @ Z))

    #denominator = np.trace(
    #                + (A.T @ Z.T @ Z @ A) \
    #                + (Z.T @ B.T @ B @ Z) \
    #                - 2*(A.T @ Z.T @ B @ Z))

    alpha = numerator/denominator

    # Ensure alpha is within [0,1].
    if (alpha > 1):
        alpha = 1
    elif (alpha < 0):
        alpha = 0

    return alpha


def optimal_ds(A, B, max_iter, tol, Q0=None, verbose=False):
    """Finds the optimal doubly stochastic matrix Q, minimising
        || Q A - B Q ||_{Fro}
    using the Frank-Wolfe algorithm.
    Inputs:
        A: A (k x k) matrix.
        B: A (k x k) matrix.
        tol: This is the tolerance on step size alpha. Any value below this and
             we assume convergence.
    """
    # Get dimensionality.
    k = A.shape[0]
    assert A.shape[1] == k
    assert B.shape[0] == k
    assert B.shape[1] == k

    # Set initial value for Q.
    if Q0 is None:
        ## Use the flat doubly stochastic matrix.
        Q0 = np.ones([k,k])/k
    else:
        assert Q0.shape[0] == k
        assert Q0.shape[1] == k

    # Can now begin main loop.
    Q = Q0.copy()
    conv = False
    grad_time = 0
    search_direction_time = 0
    step_size_time = 0
    update_time = 0
    for i in range(max_iter):
        if verbose:
            print("Iteration %s." % (i + 1))
            print("\tObjective: %s." % objective_f(A, B, Q))

        ## Compute gradient of objective at Q0.
        with Timer() as t:
            grad = compute_gradient(A, B, Q)
            grad_time += t.elapsed

        ## Compute search direction by Hungarian algorithm i.e. projecting
        ## gradient onto space of permutations/doubly stochastic matrices.
        with Timer() as t:
            W = find_search_direction(grad, k)
            search_direction_time += t.elapsed

        ## Compute step size.
        with Timer() as t:
            alpha = optimal_step_size(A, B, Q, W)
            step_size_time += t.elapsed

        if verbose:
            print("\tStep Size: %s." % alpha)

        ## Update Q.
        with Timer() as t:
            Q = (1 - alpha)*Q + alpha*W
            update_time += t.elapsed

        # Check convergence.
        if alpha < tol:
            conv = True
            break

    if verbose:
        print("Time in gradient calculation        : %s" % grad_time)
        print("Time in search direction calculation: %s" % search_direction_time)
        print("Time in step size calculation       : %s" % step_size_time)
        print("Time in update calculation          : %s" % update_time)

    if not conv:
        print("Warning, hit maximum iteration in optimal_ds.")

    return Q



