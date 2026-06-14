#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `SIP`_.

.. _SIP: https://github.com/joaospinto/sip_python

SIP is a general NLP solver based. It is based on the barrier augmented
Lagrangian method, which combines the interior point and augmented Lagrangian
methods. If you are using SIP in a scientific work, consider citing the
corresponding GitHub repository (or paper, if one has been released).

**Warm-start:** this solver interface supports warm starting 🔥
"""

import time
import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
import sip_python as sip

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def _ruiz_equilibration(P, q, A, b, G, h, num_steps):
    """Apply simple Ruiz equilibration to QP data."""
    n = q.shape[0]
    m_a = A.shape[0]
    m_g = G.shape[0]

    d = np.ones(n)
    e_a = np.ones(m_a)
    e_g = np.ones(m_g)

    P_s = P.copy()
    A_s = A.copy()
    G_s = G.copy()

    for _ in range(num_steps):
        P_abs = spa.csc_matrix(P_s.copy())
        P_abs.data = np.abs(P_abs.data)
        col_norm_x = np.array(P_abs.max(axis=0).todense()).ravel()

        if m_a > 0:
            A_abs = spa.csc_matrix(A_s.copy())
            A_abs.data = np.abs(A_abs.data)
            col_norm_a = np.array(A_abs.max(axis=0).todense()).ravel()
            col_norm_x = np.maximum(col_norm_x, col_norm_a)
            row_norm_a = np.array(A_abs.max(axis=1).todense()).ravel()
        else:
            row_norm_a = np.ones(0)

        if m_g > 0:
            G_abs = spa.csc_matrix(G_s.copy())
            G_abs.data = np.abs(G_abs.data)
            col_norm_g = np.array(G_abs.max(axis=0).todense()).ravel()
            col_norm_x = np.maximum(col_norm_x, col_norm_g)
            row_norm_g = np.array(G_abs.max(axis=1).todense()).ravel()
        else:
            row_norm_g = np.ones(0)

        safe_col = np.maximum(col_norm_x, 1e-300)
        d_step = np.where(col_norm_x > 0.0, 1.0 / np.sqrt(safe_col), 1.0)
        safe_row_a = np.maximum(row_norm_a, 1e-300)
        safe_row_g = np.maximum(row_norm_g, 1e-300)
        e_a_step = np.where(
            row_norm_a > 0.0, 1.0 / np.sqrt(safe_row_a), 1.0
        )
        e_g_step = np.where(
            row_norm_g > 0.0, 1.0 / np.sqrt(safe_row_g), 1.0
        )

        d *= d_step
        e_a *= e_a_step
        e_g *= e_g_step

        D = spa.diags(d_step)
        P_s = D @ P_s @ D
        if m_a > 0:
            A_s = spa.diags(e_a_step) @ A_s @ D
        if m_g > 0:
            G_s = spa.diags(e_g_step) @ G_s @ D

    q_s = d * q
    b_s = e_a * b if m_a > 0 else b
    h_s = e_g * h if m_g > 0 else h

    return (
        spa.csc_matrix((P_s + P_s.T) / 2.0),
        q_s,
        spa.csr_matrix(A_s),
        b_s,
        spa.csr_matrix(G_s),
        h_s,
        d,
        e_a,
        e_g,
    )


def sip_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    allow_non_psd_P: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using SIP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Notes
    -----
    ``eps_abs`` is mapped to SIP's absolute termination checks:
    ``termination.max_dual_residual``,
    ``termination.max_constraint_violation``,
    ``termination.max_complementarity_gap`` and
    ``termination.max_duality_gap``. SIP does not expose a relative QP
    termination tolerance, so ``eps_rel`` is ignored.

    Other keyword arguments are forwarded to SIP settings. Top-level SIP
    settings are passed as keyword arguments, while nested setting groups are
    passed as dictionaries. For instance:

    .. code-block:: python

       sip_solve_qp(
           P,
           q,
           G,
           h,
           max_iterations=100,
           termination={"max_dual_residual": 1e-8},
           line_search={"max_iterations": 1000},
       )

    Available SIP settings are:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Group
         - Fields
       * - top-level
         - ``max_iterations``, ``num_iterative_refinement_steps``,
           ``assert_checks_pass``
       * - ``barrier``
         - ``initial_mu``, ``mu_update_factor``, ``mu_min``,
           ``mu_update_kappa``
       * - ``penalty``
         - ``initial_penalty_parameter``,
           ``min_acceptable_constraint_violation_ratio``,
           ``penalty_parameter_increase_factor``,
           ``penalty_parameter_decrease_factor``,
           ``max_penalty_parameter``
       * - ``termination``
         - ``max_dual_residual``, ``max_constraint_violation``,
           ``max_complementarity_gap``, ``max_duality_gap``,
           ``enable_cost_change_termination``, ``max_cost_change``,
           ``max_relative_cost_change``,
           ``max_suboptimal_constraint_violation``, ``max_merit_slope``
       * - ``regularization``
         - ``initial``, ``first_positive``, ``maximum``, ``max_attempts``,
           ``increase_factor``, ``decrease_factor``
       * - ``line_search``
         - ``max_iterations``, ``tau``, ``start_ls_with_alpha_s_max``,
           ``armijo_factor``, ``line_search_factor``,
           ``line_search_min_step_size``,
           ``min_merit_slope_to_skip_line_search``, ``skip_line_search``,
           ``enable_line_search_failures``
       * - ``logging``
         - ``print_logs``, ``print_line_search_logs``,
           ``print_search_direction_logs``, ``print_derivative_check_logs``,
           ``only_check_search_direction_slope``
    Check the `Settings` struct in the `solver code
    <https://github.com/joaospinto/sip/blob/main/sip/types.hpp>`__ for details.
    """
    build_start_time = time.perf_counter()
    P, q, G_, h, A_, b, lb, ub = problem.unpack()
    original_lb = lb
    original_ub = ub
    if lb is not None and not np.any(np.isfinite(lb)):
        lb = None
    if ub is not None and not np.any(np.isfinite(ub)):
        ub = None
    if lb is not None or ub is not None:
        G_, h = linear_from_box_inequalities(
            G_, h, lb, ub, use_sparse=problem.has_sparse
        )
    n: int = q.shape[0]

    # SIP does not support A, b, G, and h to be None.
    G: Union[np.ndarray, spa.csc_matrix, spa.csr_matrix] = (
        G_ if G_ is not None else spa.csr_matrix(np.zeros((0, n)))
    )
    A: Union[np.ndarray, spa.csc_matrix, spa.csr_matrix] = (
        A_ if A_ is not None else spa.csr_matrix(np.zeros((0, n)))
    )
    h = np.zeros((0,)) if h is None else h
    b = np.zeros((0,)) if b is None else b

    # Remove vacuous inequalities before creating SIP slack variables.
    h_fin_mask = np.isfinite(h)
    if not np.all(h_fin_mask):
        G = G[h_fin_mask]
        h = h[h_fin_mask]

    if not isinstance(P, spa.csr_matrix):
        P = spa.csc_matrix(P)
        if verbose:
            warnings.warn("Converted P to a csc_matrix.")
    if not isinstance(G, spa.csr_matrix):
        G = spa.csr_matrix(G)
        if verbose:
            warnings.warn("Converted G to a csr_matrix.")
    if not isinstance(A, spa.csr_matrix):
        A = spa.csr_matrix(A)
        if verbose:
            warnings.warn("Converted A to a csr_matrix.")

    P.eliminate_zeros()
    G.eliminate_zeros()
    A.eliminate_zeros()

    ruiz_steps = kwargs.pop("ruiz_equilibration_steps", 0)
    d_scale = None
    e_a_scale = None
    e_g_scale = None
    P_unscaled = P
    q_unscaled = q
    if ruiz_steps > 0:
        P, q, A, b, G, h, d_scale, e_a_scale, e_g_scale = (
            _ruiz_equilibration(P, q, A, b, G, h, ruiz_steps)
        )

    P_T = spa.csc_matrix(P.T)
    if (
        (P.indices != P_T.indices).any()
        or (P.indptr != P_T.indptr).any()
        or (P.data != P_T.data).any()
    ):
        raise ProblemError("P should be symmetric.")

    if G is None and h is not None:
        raise ProblemError(
            "Inconsistent inequalities: G is not set but h is set"
        )
    if G is not None and h is None:
        raise ProblemError("Inconsistent inequalities: G is set but h is None")
    if A is None and b is not None:
        raise ProblemError(
            "Inconsistent inequalities: A is not set but b is set"
        )
    if A is not None and b is None:
        raise ProblemError("Inconsistent inequalities: A is set but b is None")

    k = None
    if allow_non_psd_P:
        eigenvalues, _eigenvectors = spa.linalg.eigsh(P, k=1, which="SM")
        k = -min(eigenvalues[0], 0.0) + 1e-3
    else:
        k = 1e-6

    # hess_L = P + k * spa.eye(n);
    # the code below avoids potential index cancellations.
    hess_L = spa.coo_matrix(P)
    upp_hess_L_rows = np.concatenate([hess_L.row, np.arange(n)])
    upp_hess_L_cols = np.concatenate([hess_L.col, np.arange(n)])
    upp_hess_L_data = np.concatenate([hess_L.data, k * np.ones(n)])
    hess_L = spa.coo_matrix(
        (upp_hess_L_data, (upp_hess_L_rows, upp_hess_L_cols)), shape=P.shape
    )
    hess_L.sum_duplicates()
    upp_hess_L = spa.triu(hess_L.tocsc())

    qs = sip.QDLDLSettings()
    qs.permute_kkt_system = True
    qs.kkt_pinv, kkt_nnz, kkt_L_nnz = sip.get_kkt_perm_inv_and_nnzs(
        P=hess_L,
        A=A,
        G=G,
    )

    pd = sip.ProblemDimensions()
    pd.x_dim = n
    pd.s_dim = h.shape[0]
    pd.y_dim = b.shape[0]
    pd.upper_hessian_lagrangian_nnz = upp_hess_L.nnz
    pd.jacobian_c_nnz = A.nnz
    pd.jacobian_g_nnz = G.nnz
    pd.kkt_nnz = kkt_nnz
    pd.kkt_L_nnz = kkt_L_nnz
    pd.is_jacobian_c_transposed = True
    pd.is_jacobian_g_transposed = True

    vars_ = sip.Variables(pd)

    if initvals is not None:
        if d_scale is not None:
            vars_.x[:] = initvals / d_scale  # type: ignore[index]
        else:
            vars_.x[:] = initvals  # type: ignore[index]
    else:
        vars_.x[:] = 0.0  # type: ignore[index]

    vars_.s[:] = 1.0  # type: ignore[index]
    vars_.y[:] = 0.0  # type: ignore[index]
    vars_.z[:] = 1.0  # type: ignore[index]

    ss = sip.Settings()
    ss.max_iterations = 10000
    ss.line_search.max_iterations = 200000
    ss.termination.max_dual_residual = 1e-4
    ss.termination.max_constraint_violation = 1e-4
    ss.termination.max_complementarity_gap = 1e-4
    ss.termination.max_merit_slope = 1e-16
    ss.penalty.penalty_parameter_increase_factor = 1.5
    ss.barrier.mu_update_factor = 0.8
    ss.barrier.mu_min = 1e-16
    ss.penalty.max_penalty_parameter = 1e8
    ss.assert_checks_pass = True

    ss.logging.print_logs = verbose
    ss.logging.print_line_search_logs = verbose
    ss.logging.print_search_direction_logs = verbose
    ss.logging.print_derivative_check_logs = False

    def set_sip_setting(key, value) -> bool:
        """Set a released SIP setting from qpsolvers' keyword interface."""
        top_level_settings = {
            "max_iterations",
            "num_iterative_refinement_steps",
            "assert_checks_pass",
        }
        setting_groups_by_name = {
            "barrier": ss.barrier,
            "penalty": ss.penalty,
            "termination": ss.termination,
            "regularization": ss.regularization,
            "line_search": ss.line_search,
            "logging": ss.logging,
        }
        if key in setting_groups_by_name:
            if not isinstance(value, dict):
                if verbose:
                    warnings.warn(
                        f"Expected a dictionary for SIP setting group {key}, "
                        f"received {type(value).__name__}"
                    )
                return True
            settings = setting_groups_by_name[key]
            for nested_key, nested_value in value.items():
                if hasattr(settings, nested_key):
                    setattr(settings, nested_key, nested_value)
                elif verbose:
                    warnings.warn(
                        f"Received an undefined SIP solver setting "
                        f"{key}.{nested_key} with value {nested_value}"
                    )
            return True

        if key in top_level_settings:
            setattr(ss, key, value)
            return True
        return False

    eps_abs = kwargs.pop("eps_abs", None)
    if eps_abs is not None:
        ss.termination.max_dual_residual = eps_abs
        ss.termination.max_constraint_violation = eps_abs
        ss.termination.max_complementarity_gap = eps_abs
        ss.termination.max_duality_gap = eps_abs

    eps_rel = kwargs.pop("eps_rel", None)
    if verbose and eps_rel not in (None, 0.0):
        warnings.warn("SIP does not support eps_rel; ignoring it.")

    time_limit = kwargs.pop("time_limit", float("inf"))

    for key, value in kwargs.items():
        if not set_sip_setting(key, value):
            if verbose:
                warnings.warn(
                    f"Received an undefined solver setting {key}\
                    with value {value}"
                )

    def mc(mci: sip.ModelCallbackInput) -> sip.ModelCallbackOutput:
        mco = sip.ModelCallbackOutput()

        Px = P.T @ mci.x  # type: ignore[operator]

        mco.f = 0.5 * np.dot(Px, mci.x) + np.dot(q, mci.x)
        mco.c = A @ mci.x - b  # type: ignore[operator]
        mco.g = G @ mci.x - h  # type: ignore[operator]

        mco.gradient_f = Px + q
        mco.jacobian_c = A
        mco.jacobian_g = G
        mco.upper_hessian_lagrangian = upp_hess_L

        return mco

    solver = sip.Solver(ss, qs, pd, mc, time_limit)

    solve_start_time = time.perf_counter()
    output = solver.solve(vars_)
    solve_end_time = time.perf_counter()

    solution = Solution(problem)
    solution.extras = {"sip_output": output, "sip_vars": vars_}
    solution.found = output.exit_status in (
        sip.Status.SOLVED,
        sip.Status.SUBOPTIMAL,
    )
    x_sol = np.array(vars_.x)
    y_sol = np.array(vars_.y)
    if d_scale is not None:
        x_sol = d_scale * x_sol
        y_sol = e_a_scale * y_sol

    solution.obj = 0.5 * np.dot(
        P_unscaled.T @ x_sol,  # type: ignore[operator]
        x_sol,
    ) + np.dot(q_unscaled, x_sol)
    solution.x = x_sol
    solution.y = y_sol
    if h is not None and vars_.z is not None:
        z_sip = np.array(vars_.z)
        if e_g_scale is not None:
            z_sip = e_g_scale * z_sip
        if not np.all(h_fin_mask):
            z_full = np.zeros(len(h_fin_mask))
            z_full[h_fin_mask] = z_sip
            z_sip = z_full
        z, z_box = split_dual_linear_box(z_sip, lb, ub)
        if (
            (original_lb is not None or original_ub is not None)
            and z_box.shape != (n,)
        ):
            z_box = np.zeros(n, dtype=z_sip.dtype)
        solution.z = z
        solution.z_box = z_box
    solution.build_time = solve_start_time - build_start_time
    solution.solve_time = solve_end_time - solve_start_time
    return solution


def sip_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    allow_non_psd_P: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using SIP.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `SIP
    <https://github.com/joaospinto/sip>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
    q :
        Cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    verbose :
        Set to `True` to print out extra information.
    initvals :
        Warm-start guess vector for the primal solution.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = sip_solve_problem(
        problem,
        initvals,
        verbose,
        allow_non_psd_P,
        **kwargs,
    )
    return solution.x if solution.found else None
