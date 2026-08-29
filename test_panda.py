import numpy as np
import numpy as np
import matplotlib.pyplot as plt

data = np.load("panda_solution.npz")

#print(data.files)

x = data["x_trajectory"]
u = data["u_trajectory"]

#print("x shape:", x.shape)
#print("u shape:", u.shape)

dt = 0.02
t = np.arange(u.shape[1]) * dt

plt.figure(figsize=(10, 6))

for i in range(u.shape[0]):
    plt.plot(t, u[i, :], label=f"$u_{i+1}$")

plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.title("Control trajectory")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
