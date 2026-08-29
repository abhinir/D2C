import mujoco

model = mujoco.MjModel.from_xml_path("Models/fish.xml")
data = mujoco.MjData(model)

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)
print("raw state dimension =", model.nq + model.nv)
