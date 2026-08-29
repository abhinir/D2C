# Panda Box-iLQR MuJoCo Benchmark

This bundle creates a 7-DoF torque-controlled Franka Panda benchmark for
unconstrained iLQR vs Box-iLQR.

The robot model is based on Google DeepMind's MuJoCo Menagerie
`franka_emika_panda` model. The setup script downloads the current upstream
model and assets, preserves the upstream license, freezes the gripper joints,
adds an end-effector site, and replaces the arm position servos with seven
direct torque motors.

## Why this version is useful

The MuJoCo plant does not clip torque commands. Therefore the exact same
dynamics model can be used for both:

- unconstrained iLQR: no optimizer torque bounds;
- Box-iLQR: impose `u_min <= u <= u_max` inside Box-iLQR.

Recommended physical torque limits:

```text
joint 1: +/-87 Nm
joint 2: +/-87 Nm
joint 3: +/-87 Nm
joint 4: +/-87 Nm
joint 5: +/-12 Nm
joint 6: +/-12 Nm
joint 7: +/-12 Nm
```

## 1. Install Python requirements

```bash
pip install -r requirements.txt
```

On macOS, use `mjpython` whenever the interactive MuJoCo viewer is opened.

## 2. Download and build the torque Panda model

From the bundle directory:

```bash
python setup_panda_model.py
```

For final paper reproducibility, pin the Menagerie source to a commit SHA:

```bash
MENAGERIE_REF=<commit-sha> python setup_panda_model.py
```

This creates:

```text
Models/Panda/
├── assets/
├── LICENSE
├── panda_menagerie_original.xml
├── panda_torque.xml
└── panda_box_scene.xml
```

## 3. Sanity check

```bash
python check_panda.py
```

Expected dimensions:

```text
nq = 7
nv = 7
nx = 14
nu = 7
```

The script also saves:

```text
panda_initial_guess.npz
```

## 4. Render the initial trajectory

macOS:

```bash
mjpython render_panda.py
```

Offscreen only:

```bash
python render_panda.py --no-view
```

It saves:

```text
panda_reaching.mp4
```

The default trajectory is the gravity-compensation initial guess, so it is
mainly a visualization / model test.

## 5. Use in your solver

```python
from panda_model import PandaReachModel

model = PandaReachModel(
    xml_path="Models/Panda/panda_box_scene.xml",
    horizon=100,
    n_substeps=10,
)

x0 = model.x0
u_init = model.initial_control_guess()

print(model.nx)      # 14
print(model.nu)      # 7
print(model.dt)      # 0.02 s
print(model.u_min)
print(model.u_max)
```

The dynamics call is:

```python
x_next = model.propagate_dynamics(x, u, t)
```

The running cost is:

```python
l = model.cost(x, u)
```

The terminal cost is:

```python
lf = model.terminal_cost(x)
```

The Gauss-Newton cost derivatives are:

```python
cx, cu, cxx, cuu, cux = model.quadratize_cost(x, u)
```

Terminal derivatives:

```python
cx, cu, cxx, cuu, cux = model.quadratize_cost(
    x_terminal,
    np.zeros(model.nu),
    terminal=True,
)
```

## 6. Suggested cost

Running cost:

```text
0.5 * (p_ee-p_goal)' Q_pos (p_ee-p_goal)
+ 0.5 * qdot' Q_vel qdot
+ 0.5 * u' R u
```

with:

```text
Q_pos = 100 I3
Q_vel = 0.05 I7
R = 0.10 diag(1/u_max^2)
```

Terminal cost:

```text
0.5 * (p_ee-p_goal)' Qf_pos (p_ee-p_goal)
+ 0.5 * qdot' Qf_vel qdot
```

with:

```text
Qf_pos = 5000 I3
Qf_vel = 10 I7
```

## 7. Suggested horizon

```text
MuJoCo physics dt = 0.002 s
substeps/control  = 10
control dt        = 0.02 s
horizon           = 100
final time        = 2.0 s
```

For a stronger saturation study, shorten the final time to 1.5 s or tighten
the Box-iLQR limits to 75% / 50% of nominal torque authority.

## 8. Model-free A/B estimation

`model_free_linearization.py` provides a local regression that independently
perturbs both state and control at every nominal time step.

For the Panda:

```text
nx + nu = 21
```

so use substantially more than 21 local samples. The default is 120.

It also returns rank and condition-number diagnostics.

## 9. Rendering your optimized solution

Save:

```python
np.savez(
    "panda_solution.npz",
    x_trajectory=x_trajectory,
    u_trajectory=u_trajectory,
)
```

Then:

```bash
mjpython render_panda.py --trajectory panda_solution.npz
```

or:

```bash
python render_panda.py \
    --trajectory panda_solution.npz \
    --no-view
```

## 10. Paper comparisons worth reporting

For unconstrained iLQR vs Box-iLQR, report:

- final Cartesian error;
- cost;
- maximum requested torque;
- maximum torque violation;
- fraction of controls at a box boundary;
- iterations to convergence;
- wall-clock time;
- accepted line-search alpha;
- success rate across initial configurations / targets.

A particularly clear paper figure is seven torque histories with the
+/- torque bounds drawn as dashed lines.

## Attribution

The Panda assets/model downloaded by `setup_panda_model.py` come from:

Google DeepMind, MuJoCo Menagerie:
https://github.com/google-deepmind/mujoco_menagerie

The model directory ships its own LICENSE. Preserve and follow it when
redistributing model assets.
