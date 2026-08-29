import mujoco
import mujoco.viewer
import time
import numpy as np

# The path to your XML file
xml_path = "Models/go2.xml"

# --- Load the Model and Data ---
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
except mujoco.FatalError as e:
    print(f"Error loading model: {e}")
    print("\nThis error is likely because the XML file includes other files (e.g., './common/visual.xml') that were not found.")
    print("Please ensure the 'common' directory from the MuJoCo Menagerie is in the same directory as this script.")
    exit(1)

# --- Create the Viewer and Run the Simulation ---

# Launch an interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Set the camera to a good viewing angle and distance
    viewer.cam.azimuth = 90      # Azimuthal angle (in degrees)
    viewer.cam.elevation = -15  # Elevation angle (in degrees)
    viewer.cam.distance = 5.0   # Distance from the lookat point
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0] # Point to look at (x, y, z)
    
    
#    print("\nSimulation started. Use your mouse to interact:")
#    print("  - Left-click and drag to rotate the camera.")
#    print("  - Right-click and drag to pan the camera.")
#    print("  - Scroll to zoom in and out.")
#    print("  - Ctrl + Left-click and drag on an object to apply forces (try pushing the walker!).\n")

    # The main simulation loop
    while viewer.is_running():
        step_start_time = time.time()
        
        # Advance the simulation by one step
#        data.ctrl = 10*np.ones((6))
#        data.ctrl = np.random.uniform(-10, 10, [6,])
#        print(model.qpos0)
        mujoco.mj_step(model, data)
        print(data.ctrl.shape)
#        print(data.qvel.shape)
        

        # Update the viewer with the new simulation state
        viewer.sync()

        # Maintain a real-time simulation speed
        # The timestep is defined in the XML as 0.0025s [1]
        time_until_next_step = model.opt.timestep - (time.time() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation finished.")

