from iLQR_model_free import ModelFree_ILQR
from box_iLQR_model_free import ModelFree_Box_ILQR
from models import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mujoco
import mujoco.viewer
import imageio



#model = Pendulum()
model = Fish()
#model = Pendulum(int_type = "RK4")


#control = np.loadtxt('data/controls_u.csv', delimiter=',')
control = 0*np.ones((model.nu, model.horizon))



ilqr = ModelFree_ILQR(
    model,
    max_iterations=500,
    alpha = 1,
    verbose=True
)

#ilqr.main_func(u_init=np.vstack([model.g*np.ones((1, model.horizon)),np.zeros((model.nu-1, model.horizon))]), x0 = model.x0)
ilqr.main_func(u_init=control, x0 = model.x0)
    

print(f"Minimum control: {np.min(ilqr.u_optimal)}")
print(f"Maximum control: {np.max(ilqr.u_optimal)}")
# Plot results
#save_to_csv(ilqr.x_optimal,ilqr.u_optimal,ilqr.costs)
#plot_results(ilqr.x_optimal,ilqr.u_optimal,ilqr.costs)

xml_path = model.xml_path

# --- Load the Model and Data ---
try:
    mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(mujoco_model)
except mujoco.FatalError as e:
    print(f"Error loading model: {e}")
    print("\nThis error is likely because the XML file includes other files (e.g., './common/visual.xml') that were not found.")
    print("Please ensure the 'common' directory from the MuJoCo Menagerie is in the same directory as this script.")
    exit(1)


# Create a renderer for offscreen rendering
renderer = mujoco.Renderer(mujoco_model, height=480, width=640)

# --- Simulation and Recording ---
duration = model.video_duration  # (seconds)
framerate = model.framerate # (frames per second)

# List to store video frames
frames = []

# Set initial joint positions for a more interesting start (optional)
mujoco.mj_resetData(mujoco_model, data)

# --- Create the Viewer and Run the Simulation ---

# Launch an interactive viewer
with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    
    # Set the camera to a good viewing angle and distance
    viewer.cam.azimuth = 90      # Azimuthal angle (in degrees)
    viewer.cam.elevation = 0  # Elevation angle (in degrees)
    viewer.cam.distance = 3.0   # Distance from the lookat point
    viewer.cam.lookat[:] = [0, 0.8, 0.8] # Point to look at (x, y, z)
    
    
#    print("\nSimulation started. Use your mouse to interact:")
#    print("  - Left-click and drag to rotate the camera.")
#    print("  - Right-click and drag to pan the camera.")
#    print("  - Scroll to zoom in and out.")
#    print("  - Ctrl + Left-click and drag on an object to apply forces (try pushing the walker!).\n")

    # The main simulation loop
    k = 0
    for k in range(model.horizon):
#        for i in range(10):
    #        while viewer.is_running():
        step_start_time = time.time()
        
        # Advance the simulation by one step
        for i in range(model.time_steps):
            data.ctrl = ilqr.u_optimal[:, k]
            mujoco.mj_step(mujoco_model, data)
            

            # Update the viewer with the new simulation state
            viewer.sync()
            renderer.update_scene(data, camera = viewer.cam)
            frame = renderer.render()
            frames.append(frame)

        # Maintain a real-time simulation speed
        # The timestep is defined in the XML as 0.0025s [1]
            time_until_next_step = mujoco_model.opt.timestep - (time.time() - step_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

print("Simulation finished.")
output_path = model.output_path
with imageio.get_writer(output_path, fps=framerate) as writer:
    for frame in frames:
        writer.append_data(frame)

print(f"\nVideo saved to {output_path}")
