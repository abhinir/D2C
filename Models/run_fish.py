import argparse
import platform
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================
XML_PATH = "fish.xml"
VIDEO_PATH = "fish_swimming.mp4"

SIM_DURATION = 10.0          # seconds
VIDEO_FPS = 30
WIDTH = 960
HEIGHT = 540

# Sinusoidal swimming control
TAIL_FREQUENCY = 1.5        # Hz
TAIL1_AMPLITUDE = 0.40      # rad
TAIL2_AMPLITUDE = 0.65      # rad
TAIL2_PHASE = -0.8          # rad

# "Ping" control: short pulse applied to the first tail servo
PING_AMPLITUDE = 0.45       # rad
PING_START = 1.0            # s
PING_DURATION = 0.30        # s


def swimming_control(t):
    """Continuous periodic tail motion."""
    u1 = TAIL1_AMPLITUDE * np.sin(2.0 * np.pi * TAIL_FREQUENCY * t)
    u2 = TAIL2_AMPLITUDE * np.sin(
        2.0 * np.pi * TAIL_FREQUENCY * t + TAIL2_PHASE
    )
    return np.array([u1, u2])


def ping_control(t):
    """Short pulse for testing the system response."""
    if PING_START <= t < PING_START + PING_DURATION:
        u1 = PING_AMPLITUDE
    elif PING_START + PING_DURATION <= t < PING_START + 2.0 * PING_DURATION:
        u1 = -PING_AMPLITUDE
    else:
        u1 = 0.0

    # A smaller delayed pulse on the second tail joint
    if PING_START + 0.10 <= t < PING_START + PING_DURATION + 0.10:
        u2 = 0.6 * PING_AMPLITUDE
    elif (
        PING_START + PING_DURATION + 0.10
        <= t
        < PING_START + 2.0 * PING_DURATION + 0.10
    ):
        u2 = -0.6 * PING_AMPLITUDE
    else:
        u2 = 0.0

    return np.array([u1, u2])


def make_camera(model):
    """Create a camera that follows the fish."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    fish_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "fish"
    )

    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = fish_id
    cam.distance = 2.2
    cam.azimuth = 110
    cam.elevation = -18

    return cam


def set_viewer_camera(viewer, model):
    fish_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "fish"
    )

    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = fish_id
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 110
        viewer.cam.elevation = -18


def main():
    parser = argparse.ArgumentParser(
        description="Simulate, view, and record the MuJoCo fish."
    )
    parser.add_argument(
        "--control",
        choices=["sine", "ping"],
        default="sine",
        help="Control input: periodic swimming or a short ping/pulse.",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Record without opening the interactive MuJoCo viewer.",
    )
    parser.add_argument(
        "--xml",
        default=XML_PATH,
        help="Path to the fish MJCF XML file.",
    )
    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Output MP4 path.",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Could not find {xml_path.resolve()}\n"
            "Put run_fish.py in the same folder as fish.xml, "
            "or pass --xml /path/to/fish.xml."
        )

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nu < 2:
        raise RuntimeError(
            f"The model has only {model.nu} actuator(s). "
            "This script expects tail_motor1 and tail_motor2."
        )

    # --------------------------------------------------------
    # Renderer for MP4
    # --------------------------------------------------------
    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )
    camera = make_camera(model)

    frame_period = 1.0 / VIDEO_FPS
    next_frame_time = 0.0

    controller = (
        swimming_control if args.control == "sine" else ping_control
    )

    video_path = Path(args.video)

    print(f"Model:       {xml_path.resolve()}")
    print(f"Control:     {args.control}")
    print(f"Simulation:  {SIM_DURATION:.1f} s")
    print(f"Saving to:   {video_path.resolve()}")

    # --------------------------------------------------------
    # Helper: perform one simulation step
    # --------------------------------------------------------
    def advance_one_step(writer):
        nonlocal next_frame_time

        # Position actuators: ctrl contains desired tail joint angles [rad].
        data.ctrl[:2] = controller(data.time)

        mujoco.mj_step(model, data)

        # Save frames at exactly VIDEO_FPS, independent of simulation dt.
        if data.time + 1e-12 >= next_frame_time:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            writer.append_data(frame)
            next_frame_time += frame_period

    # --------------------------------------------------------
    # Simulation + recording
    # --------------------------------------------------------
    with imageio.get_writer(
        str(video_path),
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:

        if args.no_view:
            # Fast/offscreen mode.
            while data.time < SIM_DURATION:
                advance_one_step(writer)

        else:
            # Interactive viewer + simultaneous MP4 recording.
            with mujoco.viewer.launch_passive(
                model,
                data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:

                set_viewer_camera(viewer, model)

                while viewer.is_running() and data.time < SIM_DURATION:
                    wall_start = time.time()

                    advance_one_step(writer)
                    viewer.sync()

                    # Try to keep the interactive window near real time.
                    remaining = model.opt.timestep - (time.time() - wall_start)
                    if remaining > 0:
                        time.sleep(remaining)

    renderer.close()

    print("\nFinished.")
    print(f"Video saved as: {video_path.resolve()}")


if __name__ == "__main__":
    main()
