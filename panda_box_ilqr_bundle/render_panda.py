from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from models import PandaReachModel


WIDTH = 1280
HEIGHT = 720
FPS = 30


def make_camera():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.15, 0.0, 0.50])
    cam.distance = 1.9
    cam.azimuth = 135
    cam.elevation = -22

    return cam


def apply_viewer_camera(viewer):
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.15, 0.0, 0.50])
        viewer.cam.distance = 1.9
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22


def load_trajectory(path, model):
    data = np.load(path)

    if "x_trajectory" not in data:
        raise KeyError(
            "NPZ must contain 'x_trajectory'."
        )

    x = data["x_trajectory"]

    if x.shape[0] != model.nx:
        raise ValueError(
            f"Expected x_trajectory shape ({model.nx}, N+1), "
            f"got {x.shape}."
        )

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="Models/Panda/panda_box_scene.xml",
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help=(
            "Optional .npz containing x_trajectory. "
            "If omitted, renders the gravity-compensation initial rollout."
        ),
    )
    parser.add_argument(
        "--video",
        default="panda_reaching.mp4",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
    )

    args = parser.parse_args()

    model = PandaReachModel(xml_path=args.xml)

    if args.trajectory is not None:
        x_traj = load_trajectory(args.trajectory, model)
    else:
        u0 = model.initial_control_guess()
        x_traj = model.rollout(u0)

    mj_model = model.model
    data = mujoco.MjData(mj_model)

    mj_model.vis.global_.offwidth = max(
        int(mj_model.vis.global_.offwidth),
        WIDTH,
    )
    mj_model.vis.global_.offheight = max(
        int(mj_model.vis.global_.offheight),
        HEIGHT,
    )

    renderer = mujoco.Renderer(
        mj_model,
        height=HEIGHT,
        width=WIDTH,
    )

    camera = make_camera()

    dt_control = model.dt
    total_time = (x_traj.shape[1] - 1) * dt_control

    frame_times = np.arange(
        0.0,
        total_time + 0.5 / FPS,
        1.0 / FPS,
    )

    def state_at_time(t):
        idx_float = t / dt_control
        idx0 = int(np.floor(idx_float))
        idx1 = min(idx0 + 1, x_traj.shape[1] - 1)

        if idx0 >= x_traj.shape[1] - 1:
            return x_traj[:, -1]

        beta = idx_float - idx0

        # Panda arm state is Euclidean, so simple interpolation is fine.
        return (
            (1.0 - beta) * x_traj[:, idx0]
            + beta * x_traj[:, idx1]
        )

    def set_render_state(x):
        data.qpos[:] = x[: model.nq]
        data.qvel[:] = x[model.nq :]
        mujoco.mj_forward(mj_model, data)

    with imageio.get_writer(
        args.video,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:

        if args.no_view:
            for t in frame_times:
                x = state_at_time(t)
                set_render_state(x)

                renderer.update_scene(
                    data,
                    camera=camera,
                )
                writer.append_data(renderer.render())

        else:
            with mujoco.viewer.launch_passive(
                mj_model,
                data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                apply_viewer_camera(viewer)

                for t in frame_times:
                    if not viewer.is_running():
                        break

                    start = time.time()

                    x = state_at_time(t)
                    set_render_state(x)

                    renderer.update_scene(
                        data,
                        camera=camera,
                    )
                    writer.append_data(renderer.render())

                    viewer.sync()

                    remaining = 1.0 / FPS - (time.time() - start)
                    if remaining > 0:
                        time.sleep(remaining)

    renderer.close()

    print(f"Saved video: {Path(args.video).resolve()}")


if __name__ == "__main__":
    main()
