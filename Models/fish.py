<mujoco model="simple_fish">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

    <option timestep="0.005"
            gravity="0 0 0"
            density="1000"
            viscosity="0.001"
            integrator="RK4"/>

    <visual>
        <headlight diffuse="0.8 0.8 0.8"
                   ambient="0.3 0.3 0.3"
                   specular="0.2 0.2 0.2"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>

    <asset>
        <texture name="water_tex"
                 type="2d"
                 builtin="gradient"
                 rgb1="0.05 0.15 0.22"
                 rgb2="0.00 0.03 0.07"
                 width="512"
                 height="512"/>

        <material name="water_mat"
                  texture="water_tex"
                  texrepeat="1 1"
                  reflectance="0.05"/>

        <material name="fish_body_mat"
                  rgba="0.18 0.45 0.72 1"/>

        <material name="fish_tail_mat"
                  rgba="0.12 0.32 0.55 1"/>

        <material name="fin_mat"
                  rgba="0.25 0.55 0.75 0.85"/>
    </asset>

    <worldbody>
        <!-- Visual reference floor only. Gravity is disabled. -->
        <geom name="floor"
              type="plane"
              size="6 6 0.1"
              pos="0 0 -1.5"
              material="water_mat"
              contype="0"
              conaffinity="0"/>

        <light pos="0 0 5" dir="0 0 -1"/>

        <!-- Main fish body -->
        <body name="fish" pos="0 0 0">
            <freejoint name="root"/>

            <!-- Main streamlined body.
                 Long axis is x; fish swims approximately in +x. -->
            <geom name="body_geom"
                  type="ellipsoid"
                  size="0.35 0.11 0.14"
                  pos="0 0 0"
                  density="850"
                  material="fish_body_mat"
                  friction="0.2 0.1 0.01"/>

            <!-- Head -->
            <geom name="head_geom"
                  type="ellipsoid"
                  size="0.15 0.105 0.12"
                  pos="0.28 0 0"
                  density="850"
                  material="fish_body_mat"
                  friction="0.2 0.1 0.01"/>

            <!-- Dorsal fin -->
            <geom name="dorsal_fin"
                  type="box"
                  size="0.10 0.006 0.07"
                  pos="-0.03 0 0.17"
                  euler="0 0 0"
                  density="300"
                  material="fin_mat"
                  contype="0"
                  conaffinity="0"/>

            <!-- Left pectoral fin -->
            <geom name="left_fin"
                  type="box"
                  size="0.10 0.045 0.006"
                  pos="0.10 0.12 -0.03"
                  euler="0 0 -20"
                  density="300"
                  material="fin_mat"
                  contype="0"
                  conaffinity="0"/>

            <!-- Right pectoral fin -->
            <geom name="right_fin"
                  type="box"
                  size="0.10 0.045 0.006"
                  pos="0.10 -0.12 -0.03"
                  euler="0 0 20"
                  density="300"
                  material="fin_mat"
                  contype="0"
                  conaffinity="0"/>

            <!-- First tail segment -->
            <body name="tail1" pos="-0.32 0 0">
                <joint name="tail_joint1"
                       type="hinge"
                       axis="0 0 1"
                       pos="0 0 0"
                       range="-35 35"
                       damping="0.15"
                       stiffness="0.0"
                       limited="true"/>

                <geom name="tail1_geom"
                      type="ellipsoid"
                      size="0.18 0.075 0.085"
                      pos="-0.15 0 0"
                      density="700"
                      material="fish_tail_mat"
                      friction="0.2 0.1 0.01"/>

                <!-- Second tail segment -->
                <body name="tail2" pos="-0.30 0 0">
                    <joint name="tail_joint2"
                           type="hinge"
                           axis="0 0 1"
                           pos="0 0 0"
                           range="-45 45"
                           damping="0.10"
                           stiffness="0.0"
                           limited="true"/>

                    <geom name="tail2_geom"
                          type="ellipsoid"
                          size="0.14 0.05 0.06"
                          pos="-0.11 0 0"
                          density="600"
                          material="fish_tail_mat"
                          friction="0.2 0.1 0.01"/>

                    <!-- Caudal fin -->
                    <geom name="caudal_fin"
                          type="box"
                          size="0.08 0.15 0.005"
                          pos="-0.24 0 0"
                          density="250"
                          material="fin_mat"
                          friction="0.1 0.05 0.01"/>
                </body>
            </body>

            <!-- Useful sensors/sites -->
            <site name="imu_site"
                  pos="0 0 0"
                  size="0.015"
                  rgba="1 0 0 1"/>

            <site name="nose_site"
                  pos="0.43 0 0"
                  size="0.012"
                  rgba="0 1 0 1"/>
        </body>
    </worldbody>

    <actuator>
        <!-- Position servos for the two tail joints.
             Control values are desired joint angles in radians. -->
        <position name="tail_motor1"
                  joint="tail_joint1"
                  kp="6"
                  ctrlrange="-0.55 0.55"
                  forcerange="-3 3"
                  ctrllimited="true"
                  forcelimited="true"/>

        <position name="tail_motor2"
                  joint="tail_joint2"
                  kp="4"
                  ctrlrange="-0.75 0.75"
                  forcerange="-2 2"
                  ctrllimited="true"
                  forcelimited="true"/>
    </actuator>

    <sensor>
        <framepos name="fish_position"
                  objtype="site"
                  objname="imu_site"/>

        <framequat name="fish_orientation"
                   objtype="site"
                   objname="imu_site"/>

        <framelinvel name="fish_linear_velocity"
                     objtype="site"
                     objname="imu_site"/>

        <frameangvel name="fish_angular_velocity"
                     objtype="site"
                     objname="imu_site"/>

        <jointpos name="tail1_angle"
                  joint="tail_joint1"/>

        <jointpos name="tail2_angle"
                  joint="tail_joint2"/>
    </sensor>
</mujoco>
