from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


# For final paper experiments, set MENAGERIE_REF to a specific commit SHA.
# Example:
#   MENAGERIE_REF=<commit-sha> python setup_panda_model.py
MENAGERIE_REF = os.environ.get("MENAGERIE_REF", "main")

if MENAGERIE_REF == "main":
    MENAGERIE_ZIP = (
        "https://codeload.github.com/google-deepmind/"
        "mujoco_menagerie/zip/refs/heads/main"
    )
else:
    MENAGERIE_ZIP = (
        "https://codeload.github.com/google-deepmind/"
        f"mujoco_menagerie/zip/{MENAGERIE_REF}"
    )

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "Models" / "Panda"
TMP_ZIP = HERE / "_mujoco_menagerie_main.zip"
TMP_EXTRACT = HERE / "_mujoco_menagerie_extract"


def indent_xml(tree):
    # Python >= 3.9
    ET.indent(tree, space="  ")


def download_menagerie():
    print("Downloading current MuJoCo Menagerie...")
    print(MENAGERIE_ZIP)
    urllib.request.urlretrieve(MENAGERIE_ZIP, TMP_ZIP)


def extract_panda():
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)

    with zipfile.ZipFile(TMP_ZIP, "r") as zf:
        zf.extractall(TMP_EXTRACT)

    candidates = list(TMP_EXTRACT.glob("mujoco_menagerie-*/franka_emika_panda"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate the extracted Franka Panda directory. "
            f"Candidates: {candidates}"
        )

    src = candidates[0]

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    shutil.copytree(src, MODEL_DIR)

    original = MODEL_DIR / "panda.xml"
    backup = MODEL_DIR / "panda_menagerie_original.xml"
    shutil.copy2(original, backup)

    return original


def patch_to_torque_model(original_xml: Path):
    tree = ET.parse(original_xml)
    root = tree.getroot()

    # ------------------------------------------------------------------
    # Keep the original Panda mechanics, but use a fixed gripper so that
    # the optimization state is exactly:
    #   q in R^7, qdot in R^7  -> nx = 14
    # ------------------------------------------------------------------
    for body_name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{body_name}']")
        if body is not None:
            for child in list(body):
                if child.tag == "joint":
                    body.remove(child)

    # These sections reference the two finger joints / original actuators.
    for tag in ("tendon", "equality", "actuator", "keyframe"):
        for elem in list(root.findall(tag)):
            root.remove(elem)

    # ------------------------------------------------------------------
    # Add an end-effector site at the gripper center.
    # ------------------------------------------------------------------
    hand = root.find(".//body[@name='hand']")
    if hand is None:
        raise RuntimeError("Could not find Panda hand body.")

    existing_site = hand.find("site[@name='ee_site']")
    if existing_site is None:
        ET.SubElement(
            hand,
            "site",
            {
                "name": "ee_site",
                "pos": "0 0 0.105",
                "size": "0.012",
                "rgba": "0.15 0.9 0.25 1",
            },
        )

    # ------------------------------------------------------------------
    # Set a definite physics timestep for reproducibility.
    # Menagerie already uses implicitfast; retain that integrator.
    # ------------------------------------------------------------------
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("integrator", "implicitfast")
    option.set("timestep", "0.002")

    # ------------------------------------------------------------------
    # Direct torque actuators.
    #
    # IMPORTANT:
    # No ctrlrange and ctrllimited=false here. This keeps the MuJoCo
    # plant identical for:
    #   (1) unconstrained iLQR, and
    #   (2) Box-iLQR.
    #
    # Box-iLQR should enforce the torque box limits in the optimizer.
    # ------------------------------------------------------------------
    actuator = ET.Element("actuator")

    for i in range(1, 8):
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"tau{i}",
                "joint": f"joint{i}",
                "gear": "1",
                "ctrllimited": "false",
                "forcelimited": "false",
            },
        )

    # Insert before contact if available; otherwise append.
    contact = root.find("contact")
    if contact is not None:
        index = list(root).index(contact)
        root.insert(index, actuator)
    else:
        root.append(actuator)

    torque_xml = MODEL_DIR / "panda_torque.xml"
    indent_xml(tree)
    tree.write(
        torque_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    return torque_xml


def make_scene():
    scene = MODEL_DIR / "panda_box_scene.xml"

    scene.write_text(
        """<mujoco model="panda_box_ilqr_scene">
  <include file="panda_torque.xml"/>

  <statistic center="0.2 0 0.5" extent="1.2"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7"
               ambient="0.35 0.35 0.35"
               specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.20 0.25 1"/>
    <global azimuth="135"
            elevation="-20"
            offwidth="1280"
            offheight="720"/>
  </visual>

  <asset>
    <texture name="skybox"
             type="skybox"
             builtin="gradient"
             rgb1="0.28 0.40 0.55"
             rgb2="0.02 0.03 0.05"
             width="512"
             height="3072"/>

    <texture name="groundplane"
             type="2d"
             builtin="checker"
             mark="edge"
             rgb1="0.20 0.22 0.25"
             rgb2="0.10 0.12 0.15"
             markrgb="0.8 0.8 0.8"
             width="300"
             height="300"/>

    <material name="groundplane"
              texture="groundplane"
              texuniform="true"
              texrepeat="5 5"
              reflectance="0.15"/>

    <material name="target_material"
              rgba="0.95 0.18 0.12 0.85"/>
  </asset>

  <worldbody>
    <geom name="ground"
          type="plane"
          size="2 2 0.1"
          material="groundplane"
          pos="0 0 0"
          contype="0"
          conaffinity="0"/>

    <!-- Reachable Cartesian target for the end-effector site. -->
    <geom name="target"
          type="sphere"
          pos="0.45 0.20 0.55"
          size="0.035"
          material="target_material"
          contype="0"
          conaffinity="0"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    return scene


def cleanup():
    if TMP_ZIP.exists():
        TMP_ZIP.unlink()
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)


def main():
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_menagerie()
        original = extract_panda()
        torque_xml = patch_to_torque_model(original)
        scene = make_scene()

        print("\nCreated:")
        print(f"  {torque_xml}")
        print(f"  {scene}")
        print(f"  {MODEL_DIR / 'assets'}")
        print(f"  {MODEL_DIR / 'LICENSE'}")
        print("\nThe original upstream XML is preserved as:")
        print(f"  {MODEL_DIR / 'panda_menagerie_original.xml'}")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
