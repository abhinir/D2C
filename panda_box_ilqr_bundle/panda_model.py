from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class PandaReachModel:
    """
    7-DoF torque-controlled Franka Panda reaching model.

    State:
        x = [q; qdot] in R^14

    Control:
        u = joint torques in R^7

    MuJoCo itself does NOT clip torque commands in panda_torque.xml.
    Use u_min/u_max inside Box-iLQR.
    """

    def __init__(
        self,
        xml_path="Models/Panda/panda_box_scene.xml",
        horizon=100,
        n_substeps=10,
    ):
        self.xml_path = str(Path(xml_path))

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Separate data object so cost evaluation does not overwrite
        # the dynamics state.
        self.cost_data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.model.nu

        if (self.nq, self.nv, self.nu) != (7, 7, 7):
            raise RuntimeError(
                "Expected torque Panda dimensions nq=7, nv=7, nu=7, "
                f"but got nq={self.nq}, nv={self.nv}, nu={self.nu}. "
                "Run setup_panda_model.py first."
            )

        self.horizon = int(horizon)
        self.n_substeps = int(n_substeps)

        # panda_torque.xml uses a 0.002 s physics timestep.
        self.dt_physics = float(self.model.opt.timestep)
        self.dt = self.dt_physics * self.n_substeps

        # --------------------------------------------------------------
        # Physical torque limits used by Box-iLQR.
        # The MuJoCo plant itself is intentionally left unclipped.
        # --------------------------------------------------------------
        self.u_max = np.array(
            [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            dtype=float,
        )
        self.u_min = -self.u_max

        # --------------------------------------------------------------
        # Initial configuration: based on the Menagerie home pose.
        # --------------------------------------------------------------
        self.q0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.57079,
                0.0,
                1.57079,
                -0.7853,
            ],
            dtype=float,
        )
        self.qd0 = np.zeros(7)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q0
        self.data.qvel[:] = self.qd0
        mujoco.mj_forward(self.model, self.data)

        self.x0 = np.concatenate(
            (self.data.qpos.copy(), self.data.qvel.copy())
        )

        # --------------------------------------------------------------
        # IDs.
        # --------------------------------------------------------------
        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "ee_site",
        )
        self.target_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "target",
        )

        if self.ee_site_id < 0:
            raise RuntimeError("ee_site not found.")
        if self.target_geom_id < 0:
            raise RuntimeError("target geom not found.")

        # Target is fixed in world coordinates.
        self.goal_pos = self.model.geom_pos[
            self.target_geom_id
        ].copy()

        # --------------------------------------------------------------
        # Cost.
        #
        # Position is the primary task.
        # Velocity penalty encourages arrival at rest.
        #
        # Control penalty is normalized by torque authority so the
        # 12 Nm wrist joints and 87 Nm proximal joints are comparable.
        # --------------------------------------------------------------
        self.Q_pos = 100.0 * np.eye(3)
        self.Q_vel = 0.05 * np.eye(7)

        control_weight = 0.10
        self.R = control_weight * np.diag(
            1.0 / (self.u_max ** 2)
        )

        self.Qf_pos = 5000.0 * np.eye(3)
        self.Qf_vel = 10.0 * np.eye(7)

    # ------------------------------------------------------------------
    # State / kinematics helpers
    # ------------------------------------------------------------------

    def set_state(self, data, x):
        x = np.asarray(x, dtype=float)

        if x.shape != (self.nx,):
            raise ValueError(
                f"x must have shape ({self.nx},), got {x.shape}"
            )

        data.qpos[:] = x[: self.nq]
        data.qvel[:] = x[self.nq :]
        mujoco.mj_forward(self.model, data)

    def end_effector_position(self, x):
        self.set_state(self.cost_data, x)
        return self.cost_data.site_xpos[self.ee_site_id].copy()

    def position_error(self, x):
        return self.end_effector_position(x) - self.goal_pos

    # ------------------------------------------------------------------
    # Discrete dynamics
    # ------------------------------------------------------------------

    def propagate_dynamics(self, x, u, t=None):
        """
        One control interval:
            x_{k+1} = f(x_k, u_k)

        u is held constant for n_substeps MuJoCo physics steps.

        IMPORTANT:
        No clipping is performed here. Standard iLQR can be unconstrained;
        Box-iLQR should impose self.u_min <= u <= self.u_max itself.
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.shape != (self.nu,):
            raise ValueError(
                f"u must have shape ({self.nu},), got {u.shape}"
            )

        self.data.qpos[:] = x[: self.nq]
        self.data.qvel[:] = x[self.nq :]
        self.data.ctrl[:] = u

        if t is not None:
            # If t is an integer optimizer index:
            self.data.time = float(t) * self.dt

        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

            if (
                not np.all(np.isfinite(self.data.qpos))
                or not np.all(np.isfinite(self.data.qvel))
                or not np.all(np.isfinite(self.data.qacc))
            ):
                raise FloatingPointError(
                    "MuJoCo became unstable during propagation."
                )

        return np.concatenate(
            (
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            )
        )

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def cost(self, x, u):
        """
        Running cost:
          1/2 e_p' Q_pos e_p
        + 1/2 qdot' Q_vel qdot
        + 1/2 u' R u
        """
        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos
        qvel = self.cost_data.qvel

        return float(
            0.5 * e.T @ self.Q_pos @ e
            + 0.5 * qvel.T @ self.Q_vel @ qvel
            + 0.5 * u.T @ self.R @ u
        )

    def terminal_cost(self, x):
        """
        Terminal cost:
          1/2 e_p' Qf_pos e_p
        + 1/2 qdot' Qf_vel qdot
        """
        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos
        qvel = self.cost_data.qvel

        return float(
            0.5 * e.T @ self.Qf_pos @ e
            + 0.5 * qvel.T @ self.Qf_vel @ qvel
        )

    def trajectory_cost(self, x_traj, u_traj):
        J = 0.0

        for k in range(self.horizon):
            J += self.cost(
                x_traj[:, k],
                u_traj[:, k],
            )

        J += self.terminal_cost(x_traj[:, -1])

        return float(J)

    # ------------------------------------------------------------------
    # Gauss-Newton quadratization for iLQR
    # ------------------------------------------------------------------

    def quadratize_cost(self, x, u, terminal=False):
        """
        Returns:
            c_x   : (14,)
            c_u   : (7,)
            c_xx  : (14,14)
            c_uu  : (7,7)
            c_ux  : (7,14)

        For Cartesian position, uses Gauss-Newton:
            Hessian approx Jp.T @ Q @ Jp

        This avoids injecting unnecessary indefinite curvature into
        the iLQR backward pass.
        """
        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos

        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))

        mujoco.mj_jacSite(
            self.model,
            self.cost_data,
            jacp,
            jacr,
            self.ee_site_id,
        )

        if terminal:
            Qp = self.Qf_pos
            Qv = self.Qf_vel
        else:
            Qp = self.Q_pos
            Qv = self.Q_vel

        c_x = np.zeros(self.nx)
        c_xx = np.zeros((self.nx, self.nx))

        # Position part
        c_x[: self.nq] = jacp.T @ Qp @ e
        c_xx[: self.nq, : self.nq] = jacp.T @ Qp @ jacp

        # Velocity part
        qvel = self.cost_data.qvel
        c_x[self.nq :] = Qv @ qvel
        c_xx[self.nq :, self.nq :] = Qv

        if terminal:
            c_u = np.zeros(self.nu)
            c_uu = np.zeros((self.nu, self.nu))
            c_ux = np.zeros((self.nu, self.nx))
        else:
            c_u = self.R @ u
            c_uu = self.R.copy()
            c_ux = np.zeros((self.nu, self.nx))

        # Force exact numerical symmetry.
        c_xx = 0.5 * (c_xx + c_xx.T)
        c_uu = 0.5 * (c_uu + c_uu.T)

        return c_x, c_u, c_xx, c_uu, c_ux

    # ------------------------------------------------------------------
    # Initial control guess
    # ------------------------------------------------------------------

    def gravity_compensation(self, x=None):
        """
        At qdot=0, qfrc_bias is primarily gravity compensation.
        Returns a useful torque initial guess for iLQR.
        """
        if x is None:
            x = self.x0

        self.set_state(self.cost_data, x)

        # qfrc_bias contains Coriolis/centrifugal/gravity terms.
        # With qdot = 0 at x0, this is a good holding torque.
        return self.cost_data.qfrc_bias[:7].copy()

    def initial_control_guess(self):
        u_hold = self.gravity_compensation(self.x0)
        return np.repeat(
            u_hold[:, None],
            self.horizon,
            axis=1,
        )

    def rollout(self, u_traj, x0=None):
        if x0 is None:
            x0 = self.x0

        x_traj = np.zeros((self.nx, self.horizon + 1))
        x_traj[:, 0] = x0

        for k in range(self.horizon):
            x_traj[:, k + 1] = self.propagate_dynamics(
                x_traj[:, k],
                u_traj[:, k],
                k,
            )

        return x_traj
