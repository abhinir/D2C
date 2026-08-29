from __future__ import annotations

import numpy as np


def linearize_dynamics_model_free(
    model,
    x_nom,
    u_nom,
    num_rollouts=120,
    state_std=None,
    control_fraction=0.01,
    ridge=1e-8,
    rng=None,
):
    """
    Local data-driven regression:
        dx_{k+1} = A_k dx_k + B_k du_k

    Unlike a single perturbed trajectory ensemble, this perturbs x_k and u_k
    independently at EACH time step, which makes A_k identifiable at k=0 too.

    Panda has no quaternion, so ordinary Euclidean state differences are valid.

    Args:
        model:
            PandaReachModel
        x_nom:
            shape (nx, N+1)
        u_nom:
            shape (nu, N)
        num_rollouts:
            number of local samples per time step; use > nx+nu (=21)
        state_std:
            length-nx std vector. If None, a conservative default is used.
        control_fraction:
            perturbation std = control_fraction * model.u_max
        ridge:
            ridge regularization in the regression
    """
    if rng is None:
        rng = np.random.default_rng()

    nx = model.nx
    nu = model.nu
    N = model.horizon

    if num_rollouts < nx + nu:
        raise ValueError(
            f"Use at least nx+nu={nx+nu} rollouts; "
            f"got {num_rollouts}."
        )

    if state_std is None:
        # q perturbations [rad], qdot perturbations [rad/s]
        state_std = np.concatenate(
            (
                2e-4 * np.ones(model.nq),
                2e-3 * np.ones(model.nv),
            )
        )

    state_std = np.asarray(state_std, dtype=float)
    control_std = control_fraction * model.u_max

    A = np.zeros((nx, nx, N))
    B = np.zeros((nx, nu, N))

    diagnostics = []

    eye = np.eye(nx + nu)

    for k in range(N):
        DX = np.zeros((nx, num_rollouts))
        DU = np.zeros((nu, num_rollouts))
        DY = np.zeros((nx, num_rollouts))

        for p in range(num_rollouts):
            dx = rng.normal(
                loc=0.0,
                scale=state_std,
                size=nx,
            )

            du = rng.normal(
                loc=0.0,
                scale=control_std,
                size=nu,
            )

            x_p = x_nom[:, k] + dx
            u_p = u_nom[:, k] + du

            # No clipping here: the MuJoCo plant is intentionally
            # unconstrained. Box bounds belong to Box-iLQR.
            x_next_p = model.propagate_dynamics(
                x_p,
                u_p,
                k,
            )

            DX[:, p] = dx
            DU[:, p] = du
            DY[:, p] = x_next_p - x_nom[:, k + 1]

        Z = np.vstack((DX, DU))

        # Ridge least-squares:
        # P = DY Z' (Z Z' + lambda I)^-1
        gram = Z @ Z.T + ridge * eye
        P = (DY @ Z.T) @ np.linalg.solve(
            gram,
            np.eye(nx + nu),
        )

        A[:, :, k] = P[:, :nx]
        B[:, :, k] = P[:, nx:]

        rank = np.linalg.matrix_rank(Z)
        cond = np.linalg.cond(Z)

        diagnostics.append(
            {
                "k": k,
                "rank": int(rank),
                "condition": float(cond),
                "A_norm": float(np.linalg.norm(A[:, :, k])),
                "B_norm": float(np.linalg.norm(B[:, :, k])),
            }
        )

    return A, B, diagnostics
