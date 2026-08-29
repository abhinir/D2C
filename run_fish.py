import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# DEFAULT SETTINGS
# ============================================================
XML_PATH = "Models/fish.xml"
VIDEO_PATH = "fish_swimming.mp4"

SIM_DURATION = 10.0
VIDEO_FPS = 30
WIDTH = 960
HEIGHT = 540

SWIM_FREQUENCY = 2.0  # Hz


def actuator_id(model, name):
    """Return actuator ID from its MuJoCo name."""
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise ValueError(f"Actuator '{name}' not found in model.")
    return idx


def make_swim_controller(model):
    """
    Coordinated open-loop swimming motion for the official DeepMind fish.

    Actuators:
      0: tail
      1: tail_twist
      2: fins_flap
      3: finleft_pitch
      4: finright_pitch
    """
    ids = {
        "tail": actuator_id(model, "tail"),
        "tail_twist": actuator_id(model, "tail_twist"),
        "fins_flap": actuator_id(model, "fins_flap"),
        "finleft_pitch": actuator_id(model, "finleft_pitch"),
        "finright_pitch": actuator_id(model, "finright_pitch"),
    }

    def controller(t):
        w = 2.0 * np.pi * SWIM_FREQUENCY

        u = np.zeros(model.nu)

        # Main lateral tail oscillation.
        u[ids["tail"]] = 0.80 * np.sin(w * t)

        # Small tail twist, phase shifted.
        u[ids["tail_twist"]] = 0.18 * np.sin(w * t + np.pi / 2.0)

        # Pectoral fin flapping.
        u[ids["fins_flap"]] = 0.35 * np.sin(w * t + np.pi / 2.0)

        # Symmetric fin pitch.
        u[ids["finleft_pitch"]] = 0.20
        u[ids["finright_pitch"]] = 0.20

        # Respect actuator control limits in the XML.
        return np.clip(
            u,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

    return controller


def make_ping_controller(model):
    """Short pulse input useful for inspecting the transient response."""
    ids = {
        "tail": actuator_id(model, "tail"),
        "tail_twist": actuator_id(model, "tail_twist"),
        "fins_flap": actuator_id(model, "fins_flap"),
    }

    def controller(t):
        u = np.zeros(model.nu)

        if 1.00 <= t < 1.25:
            u[ids["tail"]] = 0.80
        elif 1.25 <= t < 1.50:
            u[ids["tail"]] = -0.80

        if 1.10 <= t < 1.35:
            u[ids["fins_flap"]] = 0.50
        elif 1.35 <= t < 1.60:
            u[ids["fins_flap"]] = -0.50

        if 1.00 <= t < 1.50:
            u[ids["tail_twist"]] = 0.15

        return np.clip(
            u,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

    return controller


def set_live_camera(viewer, model, camera_name):
    """Set the interactive viewer to one of the cameras already in fish.xml."""
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
    )

    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in model.")

    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def main():
    parser = argparse.ArgumentParser(
        description="Render and record the official DeepMind MuJoCo fish."
    )

    parser.add_argument(
        "--xml",
        default=XML_PATH,
        help="Path to fish.xml.",
    )

    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Output MP4 filename.",
    )

    parser.add_argument(
        "--control",
        choices=["swim", "ping", "zero"],
        default="swim",
        help="Control law to apply.",
    )

    parser.add_argument(
        "--camera",
        choices=[
            "tracking_top",
            "tracking_x",
            "tracking_y",
            "fixed_top",
            "eye",
        ],
        default="tracking_top",
        help="Camera used for viewer and video.",
    )

    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Render video without opening the interactive viewer.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=SIM_DURATION,
        help="Simulation duration in seconds.",
    )

    args = parser.parse_args()

    xml_path = Path(args.xml)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Could not find XML file:\n{xml_path.resolve()}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Avoid framebuffer-size errors when recording at 960x540.
    model.vis.global_.offwidth = max(
        model.vis.global_.offwidth, WIDTH
    )
    model.vis.global_.offheight = max(
        model.vis.global_.offheight, HEIGHT
    )

    print("\nDeepMind fish model")
    print("-------------------")
    print(f"XML: {xml_path.resolve()}")
    print(f"nq = {model.nq}")
    print(f"nv = {model.nv}")
    print(f"nu = {model.nu}")

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )
        print(f"  {i}: {name}")

    # ========================================================
    # SELECT CONTROLLER
    # ========================================================
    if args.control == "swim":
        controller = make_swim_controller(model)

    elif args.control == "ping":
        controller = make_ping_controller(model)

    else:
        controller = lambda t: np.zeros(model.nu)

    # ========================================================
    # OFFSCREEN RENDERER FOR VIDEO
    # ========================================================
    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )

    frame_period = 1.0 / VIDEO_FPS
    next_frame_time = 0.0

    video_path = Path(args.video)

    def step_and_record(writer):
        nonlocal next_frame_time

        # Apply control.
        data.ctrl[:] = controller(data.time)

        # Advance MuJoCo dynamics.
        mujoco.mj_step(model, data)

        # Record frames at VIDEO_FPS rather than every physics step.
        while data.time + 1e-12 >= next_frame_time:
            renderer.update_scene(
                data,
                camera=args.camera,
            )

            frame = renderer.render()
            writer.append_data(frame)

            next_frame_time += frame_period

    # ========================================================
    # SIMULATION
    # ========================================================
    try:
        with imageio.get_writer(
            str(video_path),
            fps=VIDEO_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:

            # ------------------------------------------------
            # OFFSCREEN-ONLY MODE
            # ------------------------------------------------
            if args.no_view:

                while data.time < args.duration:
                    step_and_record(writer)

            # ------------------------------------------------
            # LIVE VIEWER + VIDEO RECORDING
            # ------------------------------------------------
            else:

                with mujoco.viewer.launch_passive(
                    model,
                    data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:

                    set_live_camera(
                        viewer,
                        model,
                        args.camera,
                    )

                    while (
                        viewer.is_running()
                        and data.time < args.duration
                    ):

                        wall_start = time.time()

                        step_and_record(writer)

                        # Update interactive MuJoCo window.
                        viewer.sync()

                        # Approximately real-time playback.
                        sleep_time = (
                            model.opt.timestep
                            - (time.time() - wall_start)
                        )

                        if sleep_time > 0:
                            time.sleep(sleep_time)

    finally:
        renderer.close()

    print("\nFinished.")
    print(f"Video saved to:\n{video_path.resolve()}")


if __name__ == "__main__":
    main()
