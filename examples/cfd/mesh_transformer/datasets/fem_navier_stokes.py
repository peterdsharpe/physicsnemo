# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Steady incompressible Navier-Stokes reference solver (Taylor-Hood P2-P1).

This is the sibling of :mod:`fem_reference` for the program's first
solver-labeled *nonlinear* benchmark suite: it produces interior velocity and
pressure labels for the steady incompressible Navier-Stokes equations

.. math::

   -\nu\,\Delta u + (u\cdot\nabla)u + \nabla p = f, \qquad
   \nabla\cdot u = 0,

with Dirichlet velocity boundary conditions on the same polygonal (possibly
multiply-connected) domains the Laplace reference handles.  The verified
Laplace solver in :mod:`fem_reference` is deliberately left untouched; this
module reuses its meshing, P2 shape functions, and node bookkeeping.

Discretization and solve
------------------------

- **Elements.**  Taylor-Hood: P2 (quadratic Lagrange) velocity on the
  triangle's six nodes, P1 (linear) pressure on its three vertices --
  inf-sup stable, velocity :math:`O(h^3)` and pressure :math:`O(h^2)` in
  :math:`L^2` for smooth solutions.
- **Quadrature.**  7-point degree-5 Dunavant rule (weights summing to one):
  exact for the convection integrand :math:`(w\cdot\nabla u)\cdot v`
  (degree 5) and every other bilinear form here; only the manufactured
  forcing term carries a (superconvergent) quadrature error.
- **Nonlinear solve.**  Newton with the exact analytic Jacobian (both
  convection linearizations :math:`(\delta u\cdot\nabla)u` and
  :math:`(u\cdot\nabla)\delta u` assembled), a Stokes solve as initial
  guess, and a halving backtracking line search on the free-dof residual
  norm.  If Newton fails at the target viscosity, a viscosity-continuation
  ladder (:math:`8\nu \to 4\nu \to 2\nu \to \nu`, warm-started) is tried
  once before the case is declared failed -- failure is reported, never
  papered over.
- **Pressure gauge.**  The pressure is defined up to a constant; the gauge
  is fixed by a scalar Lagrange multiplier enforcing :math:`\int_\Omega p
  \,dx = 0`.  The multiplier also absorbs the (tiny, reported) discrete
  compatibility defect :math:`\oint u_h\cdot n\,ds` of the interpolated
  boundary trace: the converged multiplier satisfies
  :math:`\lambda\,|\Omega| = -\oint u_h\cdot n\,ds` and both numbers are
  diagnostics.
- **Linear solves.**  Direct sparse LU on the reduced saddle-point system
  with Dirichlet velocity dofs eliminated: MKL PARDISO (via ``pypardiso``)
  when installed, ``scipy.sparse.linalg.spsolve`` (SuperLU) otherwise --
  see :data:`LINEAR_SOLVER`.  Both reach ~1e-15 relative residuals on these
  systems; the choice affects wall time only.

Verification hooks (used by the tests and the dataset generator)
----------------------------------------------------------------

- ``forcing`` enables the method of manufactured solutions: pick analytic
  ``(u, p)``, compute :math:`f = -\nu\Delta u + (u\cdot\nabla)u + \nabla p`
  analytically (:func:`manufactured_solution`), solve with that forcing,
  compare.  The benchmark datasets themselves are generated with
  ``forcing=None`` (homogeneous momentum) -- the forcing path exists only
  for verification.
- Per-solve diagnostics include the final free-dof Newton residual, the
  weak-divergence :math:`L^2` norm (Taylor-Hood is only weakly
  divergence-free; the norm is :math:`O(h^2)` for smooth flows), the exact
  discrete boundary flux :math:`\oint u_h\cdot n\,ds` (equal to
  :math:`\int_\Omega \nabla\cdot u_h\,dx`, integrated exactly), and a global
  momentum-balance identity check: the assembled momentum residual summed
  over *all* nodes (reactions included) must equal
  :math:`\int(u\cdot\nabla)u\,dx - \int f\,dx` computed by independent
  direct quadrature -- an assembly-exactness certificate at roundoff.

This is a benchmark-local research utility, not a proposed public API.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from fem_reference import (
    _build_p2_connectivity,
    _dirichlet_nodes,
    _interpolate_vertex_trace,
    _p2_shape_gradients,
    _p2_shape_values,
    _triangulate,
    _validate_loops,
)

__all__ = [
    "NavierStokesDiagnostics",
    "NavierStokesSolution",
    "NewtonError",
    "manufactured_solution",
    "solve_navier_stokes",
]

try:  # Optional fast direct solver: PARDISO (MKL) via pypardiso.
    import pypardiso as _pypardiso
except ImportError:  # pragma: no cover - exercised where pypardiso is absent
    _pypardiso = None

LINEAR_SOLVER = "pypardiso" if _pypardiso is not None else "scipy_superlu"
"""Which sparse direct solver backend this process uses (recorded per solve).

Both backends solve the same reduced Newton systems to machine precision
(measured relative residuals ~1e-15); PARDISO is several times faster on the
Taylor-Hood saddle-point Jacobians and is used when installed.  The Newton
tolerance (1e-10 relative) sits far above either backend's roundoff, so
labels are solver-backend-independent at the reported noise floor.
"""


def _sparse_solve(matrix: scipy.sparse.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Direct sparse solve through the selected backend."""

    if _pypardiso is not None:
        csr = matrix.tocsr()
        if csr.indptr.dtype != np.int32:
            csr = scipy.sparse.csr_matrix(
                (
                    csr.data,
                    csr.indices.astype(np.int32),
                    csr.indptr.astype(np.int32),
                ),
                shape=csr.shape,
            )
        return _pypardiso.spsolve(csr, rhs)
    return scipy.sparse.linalg.spsolve(matrix.tocsc(), rhs)


# Degree-5 Dunavant quadrature on the reference triangle (7 points),
# barycentric coordinates with weights summing to one: exact for the
# degree-5 convection integrand (P2 * grad P2 * P2).
_QUAD_BARYCENTRIC, _QUAD_WEIGHTS = (
    lambda a=0.470142064105115, b=0.101286507323456: (
        np.array(
            [
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 - 2.0 * a, a, a],
                [a, 1.0 - 2.0 * a, a],
                [a, a, 1.0 - 2.0 * a],
                [1.0 - 2.0 * b, b, b],
                [b, 1.0 - 2.0 * b, b],
                [b, b, 1.0 - 2.0 * b],
            ]
        ),
        np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3),
    )
)()

# P1 gradients with respect to the reference coordinates (l1, l2); rows are
# the three vertex shape functions l0, l1, l2.
_P1_REFERENCE_GRADIENTS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


class NewtonError(RuntimeError):
    """Newton failed to converge (after backtracking and continuation)."""


@dataclass(frozen=True)
class NavierStokesDiagnostics:
    """Mesh, Newton, gauge, and balance diagnostics for one N-S solve."""

    reynolds: float
    viscosity: float
    target_h: float
    n_vertices: int
    n_triangles: int
    n_velocity_nodes: int
    n_pressure_nodes: int
    n_dirichlet_nodes: int
    newton_iterations: int
    continuation_solves: int
    backtracking_steps: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    boundary_flux: float
    lagrange_multiplier: float
    gauge_mean_pressure: float
    divergence_l2: float
    divergence_l2_normalized: float
    momentum_balance_error: float
    trace_speed_max: float
    velocity_speed_max: float
    n_queries: int
    n_queries_snapped: int
    max_snap_distance: float
    mesh_seconds: float
    assemble_seconds: float
    solve_seconds: float
    evaluate_seconds: float

    def as_dict(self) -> dict:
        """Return a JSON-serializable copy (plain Python scalars only)."""

        result: dict = {}
        for key, value in self.__dict__.items():
            result[key] = (
                int(value) if isinstance(value, (int, np.integer)) else float(value)
            )
        return result


@dataclass(frozen=True)
class NavierStokesSolution:
    """Interior velocity/pressure at the query points plus diagnostics.

    The full finite-element solution (P2 node coordinates and velocities,
    P1 vertex pressures, and the P1 sub-triangulation) is retained only when
    ``keep_mesh=True`` was requested.
    """

    velocity_query: np.ndarray  # (n_q, 2)
    pressure_query: np.ndarray  # (n_q,)
    diagnostics: NavierStokesDiagnostics
    node_points: np.ndarray | None = None
    node_velocity: np.ndarray | None = None  # (n_nodes, 2)
    vertex_pressure: np.ndarray | None = None  # (n_vertices,)
    triangles: np.ndarray | None = None


class _MeshOperators:
    """Precomputed geometry, quadrature, and constant matrices for one mesh."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.n_vertices = vertices.shape[0]
        self.connectivity, self.unique_edges = _build_p2_connectivity(
            self.n_vertices, triangles
        )
        self.n_nodes = self.n_vertices + self.unique_edges.shape[0]
        self.node_points = np.concatenate(
            (
                vertices,
                0.5
                * (
                    vertices[self.unique_edges[:, 0]]
                    + vertices[self.unique_edges[:, 1]]
                ),
            ),
            axis=0,
        )

        corner = vertices[triangles]  # (n_tri, 3, 2)
        jacobian = np.stack(
            (corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=2
        )
        determinant = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1]
            - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        if np.any(determinant <= 0.0):
            raise RuntimeError("triangulation produced a non-positive Jacobian")
        inverse = (
            np.stack(
                (
                    np.stack((jacobian[:, 1, 1], -jacobian[:, 0, 1]), axis=1),
                    np.stack((-jacobian[:, 1, 0], jacobian[:, 0, 0]), axis=1),
                ),
                axis=1,
            )
            / determinant[:, None, None]
        )
        self.areas = 0.5 * determinant
        self.corner = corner

        # Physical P2 gradients per quadrature point: (n_q, n_tri, 6, 2);
        # physical P1 gradients are constant per element: (n_tri, 3, 2).
        reference = _p2_shape_gradients(_QUAD_BARYCENTRIC)  # (n_q, 6, 2)
        self.p2_gradients = np.einsum("qab,nbc->qnac", reference, inverse)
        self.p1_gradients = np.einsum("ab,nbc->nac", _P1_REFERENCE_GRADIENTS, inverse)
        self.p2_values = _p2_shape_values(_QUAD_BARYCENTRIC)  # (n_q, 6)
        self.p1_values = _QUAD_BARYCENTRIC  # (n_q, 3): P1 shapes ARE barycentric
        self.quad_scale = _QUAD_WEIGHTS[:, None] * self.areas[None, :]  # (n_q, n_tri)
        # Physical quadrature points: (n_q, n_tri, 2).
        self.quad_points = np.einsum("qc,ncd->qnd", _QUAD_BARYCENTRIC, corner)

        n_tri = triangles.shape[0]
        self._rows6 = np.repeat(self.connectivity, 6, axis=1).reshape(-1)
        self._cols6 = np.tile(self.connectivity, (1, 6)).reshape(-1)
        self._rows36 = np.repeat(triangles, 6, axis=1).reshape(-1)
        self._cols36 = np.tile(self.connectivity, (1, 3)).reshape(-1)

        # Scalar P2 stiffness K (shared by both velocity components).
        stiffness_local = np.zeros((n_tri, 6, 6))
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            gradients = self.p2_gradients[q]
            stiffness_local += self.quad_scale[q][:, None, None] * np.einsum(
                "nab,ncb->nac", gradients, gradients
            )
        self.stiffness = scipy.sparse.coo_matrix(
            (stiffness_local.reshape(-1), (self._rows6, self._cols6)),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()

        # Divergence matrices B_j (n_vertices x n_nodes): int psi_k d_j phi_a.
        self.divergence = []
        for j in range(2):
            local = np.zeros((n_tri, 3, 6))
            for q in range(_QUAD_BARYCENTRIC.shape[0]):
                local += self.quad_scale[q][:, None, None] * (
                    self.p1_values[q][None, :, None]
                    * self.p2_gradients[q][:, None, :, j]
                )
            self.divergence.append(
                scipy.sparse.coo_matrix(
                    (local.reshape(-1), (self._rows36, self._cols36)),
                    shape=(self.n_vertices, self.n_nodes),
                ).tocsr()
            )

        # Pressure-mean vector E_k = int psi_k dx (P1: |T| / 3 per vertex).
        self.pressure_mean = np.zeros(self.n_vertices)
        np.add.at(
            self.pressure_mean,
            triangles.reshape(-1),
            np.repeat(self.areas / 3.0, 3),
        )
        self.domain_area = float(self.areas.sum())

        # P2 shape-function integrals per node: int phi_a dx (for forcing
        # row-sum checks) are not needed; forcing is integrated by quadrature.

    def elementwise_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Gather nodal velocities per element: (n_tri, 2, 6)."""

        return velocity.reshape(2, self.n_nodes)[:, self.connectivity].transpose(
            1, 0, 2
        )

    def convection_matrices(
        self, velocity: np.ndarray
    ) -> tuple[scipy.sparse.csr_matrix, list[list[scipy.sparse.csr_matrix]]]:
        """Assemble C(w) and the Newton cross terms W_ij at the state ``w``.

        ``C[a, b] = int phi_a (w . grad phi_b)`` (identical for both velocity
        components) and ``W[i][j][a, b] = int phi_a phi_b d_j w_i`` (the
        linearization of the advecting field).
        """

        u_elem = self.elementwise_velocity(velocity)  # (n_tri, 2, 6)
        n_tri = self.triangles.shape[0]
        convection_local = np.zeros((n_tri, 6, 6))
        cross_local = np.zeros((n_tri, 2, 2, 6, 6))
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            phi = self.p2_values[q]  # (6,)
            gradients = self.p2_gradients[q]  # (n_tri, 6, 2)
            w_q = np.einsum("nce,e->nc", u_elem, phi)  # (n_tri, 2)
            grad_w = np.einsum("nce,ned->ncd", u_elem, gradients)  # (n_tri, 2, 2)
            advect = np.einsum("nd,nbd->nb", w_q, gradients)  # (n_tri, 6)
            scale = self.quad_scale[q]
            convection_local += scale[:, None, None] * (
                phi[None, :, None] * advect[:, None, :]
            )
            cross_local += (
                scale[:, None, None, None, None] * grad_w[:, :, :, None, None]
            ) * np.outer(phi, phi)[None, None, None]
        convection = scipy.sparse.coo_matrix(
            (convection_local.reshape(-1), (self._rows6, self._cols6)),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()
        cross = [
            [
                scipy.sparse.coo_matrix(
                    (
                        cross_local[:, i, j].reshape(-1),
                        (self._rows6, self._cols6),
                    ),
                    shape=(self.n_nodes, self.n_nodes),
                ).tocsr()
                for j in range(2)
            ]
            for i in range(2)
        ]
        return convection, cross

    def forcing_vector(
        self, forcing: Callable[[np.ndarray], np.ndarray] | None
    ) -> np.ndarray:
        """Quadrature of ``int f_i phi_a dx``; zeros when unforced."""

        result = np.zeros((2, self.n_nodes))
        if forcing is None:
            return result.reshape(-1)
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            values = np.asarray(forcing(self.quad_points[q]), dtype=np.float64)
            if values.shape != (self.triangles.shape[0], 2):
                raise ValueError("forcing must map (n, 2) points to (n, 2) values")
            weighted = self.quad_scale[q][:, None] * values  # (n_tri, 2)
            contribution = weighted[:, :, None] * self.p2_values[q][None, None, :]
            for component in range(2):
                np.add.at(
                    result[component],
                    self.connectivity.reshape(-1),
                    contribution[:, component].reshape(-1),
                )
        return result.reshape(-1)

    def divergence_field(self, velocity: np.ndarray) -> tuple[float, float, float]:
        """Return ``(||div u||_L2, ||grad u||_L2, int div u dx)`` by quadrature."""

        u_elem = self.elementwise_velocity(velocity)
        div_sq = 0.0
        grad_sq = 0.0
        div_integral = 0.0
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            grad_w = np.einsum("nce,ned->ncd", u_elem, self.p2_gradients[q])
            divergence = grad_w[:, 0, 0] + grad_w[:, 1, 1]
            scale = self.quad_scale[q]
            div_sq += float((scale * divergence**2).sum())
            grad_sq += float((scale[:, None, None] * grad_w**2).sum())
            div_integral += float((scale * divergence).sum())
        return math.sqrt(div_sq), math.sqrt(grad_sq), div_integral

    def convective_momentum_integral(self, velocity: np.ndarray) -> np.ndarray:
        """Direct quadrature of ``int (u . grad) u dx`` (independent of COO)."""

        u_elem = self.elementwise_velocity(velocity)
        total = np.zeros(2)
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            w_q = np.einsum("nce,e->nc", u_elem, self.p2_values[q])
            grad_w = np.einsum("nce,ned->ncd", u_elem, self.p2_gradients[q])
            advect = np.einsum("nd,ncd->nc", w_q, grad_w)
            total += (self.quad_scale[q][:, None] * advect).sum(axis=0)
        return total

    def forcing_integral(
        self, forcing: Callable[[np.ndarray], np.ndarray] | None
    ) -> np.ndarray:
        """Direct quadrature of ``int f dx``; zeros when unforced."""

        total = np.zeros(2)
        if forcing is None:
            return total
        for q in range(_QUAD_BARYCENTRIC.shape[0]):
            values = np.asarray(forcing(self.quad_points[q]), dtype=np.float64)
            total += (self.quad_scale[q][:, None] * values).sum(axis=0)
        return total


def _momentum_residual(
    operators: _MeshOperators,
    viscosity: float,
    convection: scipy.sparse.csr_matrix,
    velocity: np.ndarray,
    pressure: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    """Full momentum residual over all velocity dofs (Dirichlet rows kept)."""

    n = operators.n_nodes
    residual = np.empty(2 * n)
    operator = viscosity * operators.stiffness + convection
    for component in range(2):
        block = slice(component * n, (component + 1) * n)
        residual[block] = (
            operator @ velocity[block]
            - operators.divergence[component].T @ pressure
            - force[block]
        )
    return residual


def _continuity_residual(
    operators: _MeshOperators,
    velocity: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    n = operators.n_nodes
    return (
        operators.divergence[0] @ velocity[:n]
        + operators.divergence[1] @ velocity[n:]
        + multiplier * operators.pressure_mean
    )


def _assemble_jacobian(
    operators: _MeshOperators,
    viscosity: float,
    convection: scipy.sparse.csr_matrix,
    cross: list[list[scipy.sparse.csr_matrix]] | None,
) -> scipy.sparse.csr_matrix:
    """Full (2 n_nodes + n_vertices + 1) Jacobian in CSR form.

    ``cross=None`` assembles the Stokes (or Oseen) operator without the
    advecting-field linearization.
    """

    diagonal = viscosity * operators.stiffness + convection
    blocks_uu = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            entry = diagonal if i == j else None
            if cross is not None:
                term = cross[i][j]
                entry = term if entry is None else entry + term
            blocks_uu[i][j] = entry
    e_column = scipy.sparse.csr_matrix(
        operators.pressure_mean[:, None]
    )  # (n_vertices, 1)
    system = scipy.sparse.bmat(
        [
            [
                blocks_uu[0][0],
                blocks_uu[0][1],
                -operators.divergence[0].T,
                None,
            ],
            [
                blocks_uu[1][0],
                blocks_uu[1][1],
                -operators.divergence[1].T,
                None,
            ],
            [operators.divergence[0], operators.divergence[1], None, e_column],
            [None, None, e_column.T, None],
        ],
        format="csr",
    )
    return system


def _solve_newton(
    operators: _MeshOperators,
    viscosity: float,
    trace: np.ndarray,
    dirichlet_nodes: np.ndarray,
    force: np.ndarray,
    *,
    initial: tuple[np.ndarray, np.ndarray, float] | None,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Damped Newton at one viscosity; raises :class:`NewtonError` on failure.

    Returns ``(velocity, pressure, multiplier, info)``; ``info`` carries the
    iteration count, backtracking steps, and the initial/final free-dof
    residual norms.  The convergence test is
    ``||R_free|| <= tolerance * max(1, ||R_free(Stokes)||)``.
    """

    n = operators.n_nodes
    n_pressure = operators.n_vertices
    total = 2 * n + n_pressure + 1

    is_dirichlet = np.zeros(total, dtype=bool)
    is_dirichlet[dirichlet_nodes] = True
    is_dirichlet[n + dirichlet_nodes] = True
    free = np.nonzero(~is_dirichlet)[0]

    def full_residual(
        velocity: np.ndarray,
        pressure: np.ndarray,
        multiplier: float,
        convection: scipy.sparse.csr_matrix,
    ) -> np.ndarray:
        residual = np.empty(total)
        residual[: 2 * n] = _momentum_residual(
            operators, viscosity, convection, velocity, pressure, force
        )
        residual[2 * n : 2 * n + n_pressure] = _continuity_residual(
            operators, velocity, multiplier
        )
        residual[-1] = operators.pressure_mean @ pressure
        return residual

    def unpack(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        return state[: 2 * n], state[2 * n : 2 * n + n_pressure], float(state[-1])

    state = np.zeros(total)
    state[dirichlet_nodes] = trace[:, 0]
    state[n + dirichlet_nodes] = trace[:, 1]

    if initial is None:
        # Stokes initial guess: the linear saddle-point solve.
        zero_convection = scipy.sparse.csr_matrix((n, n))
        jacobian = _assemble_jacobian(operators, viscosity, zero_convection, None)
        residual = full_residual(*unpack(state), zero_convection)
        correction = _sparse_solve(jacobian[free][:, free], -residual[free])
        state[free] += correction
    else:
        warm = np.concatenate((initial[0], initial[1], [initial[2]]))
        state[free] = warm[free]

    velocity, pressure, multiplier = unpack(state)
    convection, cross = operators.convection_matrices(velocity)
    residual = full_residual(velocity, pressure, multiplier, convection)
    residual_norm = float(np.linalg.norm(residual[free]))
    initial_norm = residual_norm
    threshold = tolerance * max(1.0, initial_norm)

    backtracking_total = 0
    for iteration in range(max_iterations):
        if not math.isfinite(residual_norm):
            raise NewtonError("Newton residual is non-finite")
        if residual_norm <= threshold:
            info = {
                "newton_iterations": iteration,
                "backtracking_steps": backtracking_total,
                "initial_residual": initial_norm,
                "final_residual": residual_norm,
            }
            return velocity, pressure, multiplier, info

        jacobian = _assemble_jacobian(operators, viscosity, convection, cross)
        step = _sparse_solve(jacobian[free][:, free], -residual[free])
        if not np.isfinite(step).all():
            raise NewtonError("Newton linear solve produced non-finite values")

        # Backtracking line search on the free-dof residual norm.
        damping = 1.0
        accepted = False
        for _ in range(12):
            candidate = state.copy()
            candidate[free] += damping * step
            velocity_c, pressure_c, multiplier_c = unpack(candidate)
            convection_c, cross_c = operators.convection_matrices(velocity_c)
            residual_c = full_residual(
                velocity_c, pressure_c, multiplier_c, convection_c
            )
            norm_c = float(np.linalg.norm(residual_c[free]))
            if math.isfinite(norm_c) and norm_c < (1.0 - 1.0e-4 * damping) * (
                residual_norm
            ):
                accepted = True
                break
            damping *= 0.5
            backtracking_total += 1
        if not accepted:
            raise NewtonError(
                f"backtracking stalled at residual {residual_norm:.3e} "
                f"(iteration {iteration})"
            )
        state = candidate
        velocity, pressure, multiplier = velocity_c, pressure_c, multiplier_c
        convection, cross = convection_c, cross_c
        residual, residual_norm = residual_c, norm_c

    raise NewtonError(
        f"Newton did not converge in {max_iterations} iterations "
        f"(residual {residual_norm:.3e}, threshold {threshold:.3e})"
    )


def _evaluate_taylor_hood(
    operators: _MeshOperators,
    velocity: np.ndarray,
    pressure: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Locate queries; evaluate P2 velocity and P1 pressure; snap outsiders."""

    import matplotlib.tri as mtri
    import scipy.spatial

    vertices, triangles = operators.vertices, operators.triangles
    triangulation = mtri.Triangulation(
        vertices[:, 0], vertices[:, 1], triangles=triangles
    )
    finder = triangulation.get_trifinder()
    located = np.asarray(
        finder(
            np.ascontiguousarray(query_points[:, 0]),
            np.ascontiguousarray(query_points[:, 1]),
        ),
        dtype=np.int64,
    )

    missing = np.nonzero(located < 0)[0]
    n_snapped = int(missing.shape[0])
    max_snap = 0.0
    if n_snapped:
        vertex_to_triangle = np.full(vertices.shape[0], -1, dtype=np.int64)
        flat = triangles.reshape(-1)
        vertex_to_triangle[flat[::-1]] = np.repeat(
            np.arange(triangles.shape[0])[::-1], 3
        )
        tree = scipy.spatial.cKDTree(vertices)
        _, nearest_vertex = tree.query(query_points[missing])
        located[missing] = vertex_to_triangle[nearest_vertex]
        if np.any(located < 0):
            raise RuntimeError("query point location failed")

    corner = vertices[triangles[located]]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    rhs = query_points - corner[:, 0]
    determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    l1 = (rhs[:, 0] * edge2[:, 1] - rhs[:, 1] * edge2[:, 0]) / determinant
    l2 = (edge1[:, 0] * rhs[:, 1] - edge1[:, 1] * rhs[:, 0]) / determinant
    barycentric = np.stack((1.0 - l1 - l2, l1, l2), axis=1)
    if n_snapped:
        clipped = np.clip(barycentric[missing], 0.0, None)
        clipped /= clipped.sum(axis=1, keepdims=True)
        barycentric[missing] = clipped
        snapped_points = np.einsum("nc,ncd->nd", clipped, corner[missing])
        max_snap = float(
            np.linalg.norm(snapped_points - query_points[missing], axis=1).max()
        )

    shape_p2 = _p2_shape_values(barycentric)  # (n_q, 6)
    element_nodes = operators.connectivity[located]  # (n_q, 6)
    n = operators.n_nodes
    velocity_query = np.stack(
        [
            np.einsum(
                "na,na->n", shape_p2, velocity[c * n : (c + 1) * n][element_nodes]
            )
            for c in range(2)
        ],
        axis=-1,
    )
    pressure_query = np.einsum("na,na->n", barycentric, pressure[triangles[located]])
    return velocity_query, pressure_query, n_snapped, max_snap


def solve_navier_stokes(
    boundary_loops: Sequence[np.ndarray],
    boundary_velocity: Callable[[np.ndarray], np.ndarray] | Sequence[np.ndarray],
    query_points: np.ndarray,
    *,
    viscosity: float,
    target_h: float = 0.03,
    forcing: Callable[[np.ndarray], np.ndarray] | None = None,
    max_newton_iterations: int = 25,
    newton_tolerance: float = 1.0e-10,
    continuation: bool = True,
    keep_mesh: bool = False,
) -> NavierStokesSolution:
    r"""Solve steady incompressible N-S with Dirichlet velocity BCs.

    Parameters
    ----------
    boundary_loops
        As in :func:`fem_reference.solve_dirichlet`: outer loop first, then
        hole loops; closed implicitly.
    boundary_velocity
        Either a callable mapping ``(n, 2)`` physical points to ``(n, 2)``
        velocity values (evaluated at every boundary node), or one
        ``(n_i, 2)`` per-vertex array per loop, interpolated linearly along
        the polyline (componentwise).
    query_points
        ``(n_q, 2)`` interior evaluation points.
    viscosity
        Kinematic viscosity :math:`\nu > 0` (the momentum equation is
        :math:`-\nu\Delta u + (u\cdot\nabla)u + \nabla p = f`).  With the
        benchmark's unit-peak drives and unit reference length this equals
        :math:`1/\mathrm{Re}`.
    target_h
        Interior mesh edge-length target (see :mod:`fem_reference`).
    forcing
        Optional volume force callable, ``(n, 2)`` points to ``(n, 2)``
        values -- the manufactured-solutions verification hook.  Benchmark
        labels are generated with ``forcing=None``.
    max_newton_iterations, newton_tolerance
        Newton is declared converged when the free-dof residual norm falls
        below ``newton_tolerance * max(1, ||R(Stokes initial guess)||)``.
    continuation
        When direct Newton from the Stokes guess fails, retry along the
        warm-started viscosity ladder ``8 nu -> 4 nu -> 2 nu -> nu``.
    keep_mesh
        Retain the full finite-element solution on the returned object.

    Raises
    ------
    NewtonError
        If Newton fails at the target viscosity even after continuation.
    """

    if not (math.isfinite(viscosity) and viscosity > 0.0):
        raise ValueError("viscosity must be finite and positive")
    if not (math.isfinite(target_h) and target_h > 0.0):
        raise ValueError("target_h must be finite and positive")
    loops = _validate_loops(boundary_loops)
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 2 or query_points.shape[1] != 2:
        raise ValueError("query_points must have shape (n_q, 2)")
    if not np.isfinite(query_points).all():
        raise ValueError("query_points must be finite")
    values_callable = callable(boundary_velocity)
    if not values_callable and len(boundary_velocity) != len(loops):
        raise ValueError("per-vertex boundary_velocity must provide one array per loop")

    start = time.perf_counter()
    vertices, triangles, vertex_markers, boundary_segments = _triangulate(
        loops, target_h
    )
    mesh_seconds = time.perf_counter() - start

    start = time.perf_counter()
    operators = _MeshOperators(vertices, triangles)
    dirichlet = _dirichlet_nodes(
        operators.n_vertices, vertex_markers, boundary_segments, operators.unique_edges
    )
    dirichlet_points = operators.node_points[dirichlet]
    if values_callable:
        trace = np.asarray(boundary_velocity(dirichlet_points), dtype=np.float64)
        if trace.shape != (dirichlet.shape[0], 2):
            raise ValueError(
                "boundary_velocity callable must return one (2,) value per point"
            )
    else:
        trace = np.stack(
            [
                _interpolate_vertex_trace(
                    loops,
                    [
                        np.asarray(loop_values)[:, component]
                        for loop_values in (boundary_velocity)
                    ],
                    dirichlet_points,
                )
                for component in range(2)
            ],
            axis=-1,
        )
    if not np.isfinite(trace).all():
        raise ValueError("boundary velocity evaluated to non-finite values")
    force = operators.forcing_vector(forcing)
    assemble_seconds = time.perf_counter() - start

    start = time.perf_counter()
    continuation_solves = 0
    try:
        velocity, pressure, multiplier, info = _solve_newton(
            operators,
            viscosity,
            trace,
            dirichlet,
            force,
            initial=None,
            max_iterations=max_newton_iterations,
            tolerance=newton_tolerance,
        )
    except NewtonError:
        if not continuation:
            raise
        # Warm-started viscosity ladder: 8 nu -> 4 nu -> 2 nu -> nu.
        state: tuple[np.ndarray, np.ndarray, float] | None = None
        velocity = pressure = None  # type: ignore[assignment]
        for factor in (8.0, 4.0, 2.0, 1.0):
            velocity, pressure, multiplier, info = _solve_newton(
                operators,
                factor * viscosity,
                trace,
                dirichlet,
                force,
                initial=state,
                max_iterations=max_newton_iterations,
                tolerance=newton_tolerance,
            )
            state = (velocity, pressure, multiplier)
            continuation_solves += 1
    solve_seconds = time.perf_counter() - start

    start = time.perf_counter()
    velocity_query, pressure_query, n_snapped, max_snap = _evaluate_taylor_hood(
        operators, velocity, pressure, query_points
    )
    evaluate_seconds = time.perf_counter() - start

    divergence_l2, gradient_l2, boundary_flux = operators.divergence_field(velocity)
    convection, _ = operators.convection_matrices(velocity)
    momentum_row_sum = (
        _momentum_residual(operators, viscosity, convection, velocity, pressure, force)
        .reshape(2, operators.n_nodes)
        .sum(axis=1)
    )
    balance_reference = operators.convective_momentum_integral(
        velocity
    ) - operators.forcing_integral(forcing)
    balance_scale = max(
        float(np.linalg.norm(balance_reference)),
        viscosity * gradient_l2,
        1.0e-30,
    )
    momentum_balance_error = float(
        np.linalg.norm(momentum_row_sum - balance_reference) / balance_scale
    )
    gauge_mean_pressure = float(
        operators.pressure_mean @ pressure / operators.domain_area
    )

    trace_speed = np.linalg.norm(trace, axis=1)
    node_speed = np.linalg.norm(velocity.reshape(2, operators.n_nodes).T, axis=1)
    reynolds = float(trace_speed.max() / viscosity) if trace_speed.size else 0.0

    diagnostics = NavierStokesDiagnostics(
        reynolds=reynolds,
        viscosity=float(viscosity),
        target_h=float(target_h),
        n_vertices=int(operators.n_vertices),
        n_triangles=int(triangles.shape[0]),
        n_velocity_nodes=int(operators.n_nodes),
        n_pressure_nodes=int(operators.n_vertices),
        n_dirichlet_nodes=int(dirichlet.shape[0]),
        newton_iterations=int(info["newton_iterations"]),
        continuation_solves=int(continuation_solves),
        backtracking_steps=int(info["backtracking_steps"]),
        initial_residual=float(info["initial_residual"]),
        final_residual=float(info["final_residual"]),
        relative_residual=float(
            info["final_residual"] / max(1.0, info["initial_residual"])
        ),
        boundary_flux=float(boundary_flux),
        lagrange_multiplier=float(multiplier),
        gauge_mean_pressure=gauge_mean_pressure,
        divergence_l2=float(divergence_l2),
        divergence_l2_normalized=float(divergence_l2 / max(gradient_l2, 1.0e-30)),
        momentum_balance_error=momentum_balance_error,
        trace_speed_max=float(trace_speed.max()) if trace_speed.size else 0.0,
        velocity_speed_max=float(node_speed.max()),
        n_queries=int(query_points.shape[0]),
        n_queries_snapped=n_snapped,
        max_snap_distance=max_snap,
        mesh_seconds=mesh_seconds,
        assemble_seconds=assemble_seconds,
        solve_seconds=solve_seconds,
        evaluate_seconds=evaluate_seconds,
    )
    return NavierStokesSolution(
        velocity_query=velocity_query,
        pressure_query=pressure_query,
        diagnostics=diagnostics,
        node_points=operators.node_points if keep_mesh else None,
        node_velocity=(velocity.reshape(2, operators.n_nodes).T if keep_mesh else None),
        vertex_pressure=pressure if keep_mesh else None,
        triangles=triangles if keep_mesh else None,
    )


def manufactured_solution(viscosity: float) -> dict[str, Callable]:
    r"""Analytic (u, p) and the matching N-S forcing for MMS verification.

    The velocity derives from the streamfunction
    :math:`\psi = \sin(a(x - x_0))\,\sin(a(y - y_0))` (hence exactly
    divergence-free), the pressure is :math:`p = \cos(a(x - x_0))\,
    \sin(a(y - y_0))`, and the forcing is the analytically evaluated

    .. math::

       f = -\nu\,\Delta u + (u\cdot\nabla)u + \nabla p .

    The offsets make the trace inhomogeneous on generic domains, so the MMS
    exercises the full Dirichlet path.  Returns callables ``velocity``,
    ``pressure``, ``forcing`` mapping ``(n, 2)`` points to values.
    """

    a = math.pi
    x0, y0 = 0.31, -0.17

    def parts(points: np.ndarray):
        x = np.asarray(points, dtype=np.float64)[:, 0] - x0
        y = np.asarray(points, dtype=np.float64)[:, 1] - y0
        return (
            np.sin(a * x),
            np.cos(a * x),
            np.sin(a * y),
            np.cos(a * y),
        )

    def velocity(points: np.ndarray) -> np.ndarray:
        sx, cx, sy, cy = parts(points)
        return np.stack((a * sx * cy, -a * cx * sy), axis=-1)

    def pressure(points: np.ndarray) -> np.ndarray:
        sx, cx, sy, cy = parts(points)
        return cx * sy

    def forcing(points: np.ndarray) -> np.ndarray:
        sx, cx, sy, cy = parts(points)
        u1 = a * sx * cy
        u2 = -a * cx * sy
        # First derivatives of the velocity.
        du1_dx = a * a * cx * cy
        du1_dy = -a * a * sx * sy
        du2_dx = a * a * sx * sy
        du2_dy = -a * a * cx * cy
        # The velocity components are Laplace eigenfunctions:
        # Delta u = -2 a^2 u.
        laplacian_u1 = -2.0 * a * a * u1
        laplacian_u2 = -2.0 * a * a * u2
        dp_dx = -a * sx * sy
        dp_dy = a * cx * cy
        f1 = -viscosity * laplacian_u1 + u1 * du1_dx + u2 * du1_dy + dp_dx
        f2 = -viscosity * laplacian_u2 + u1 * du2_dx + u2 * du2_dy + dp_dy
        return np.stack((f1, f2), axis=-1)

    return {"velocity": velocity, "pressure": pressure, "forcing": forcing}
