"""FFW-SG2 / NIST model assembled from the same source assets as the Isaac scene."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).parent / "assets"
LAYOUT = json.loads((ASSETS / "layout_v0.json").read_text())
BOARD = np.array([.45, -.10, .756])
TABLE_Z = .727
PEG_PARTS = [name for name in LAYOUT["parts"] if name.startswith("peg_")]


def text(values):
    return " ".join(f"{float(v):.10g}" for v in np.asarray(values).ravel())


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return np.r_[q[3], q[:3]]


def camera_rotation(eye, target):
    z = np.asarray(eye) - np.asarray(target)
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def plate_polygon(clearance_mm):
    mesh = trimesh.load(ASSETS / "nist/GMC_Laser_Plate_Virtual.STL", force="mesh")
    mesh.apply_scale(.001)
    top = mesh.vertices[:, 2].max()
    triangles = mesh.triangles[np.all(np.abs(mesh.triangles[:, :, 2] - top) < 1e-7, axis=1)]
    plate = unary_union([Polygon(t[:, :2] - .192) for t in triangles]).simplify(.000025)
    # Increase round-hole diameter by the configured training clearance. All
    # remaining holes/cut-outs come directly from the NIST plate, not its hull.
    for name in PEG_PARTS:
        part = LAYOUT["parts"][name]
        xy = part["pose_assembled"][:2]
        if "round" in name:
            radius = float(name.rsplit("_", 1)[1]) / 2000 + clearance_mm / 2000
            plate = plate.difference(Point(xy).buffer(radius, quad_segs=24))
    if not isinstance(plate, Polygon) or not plate.is_valid:
        raise ValueError("NIST plate must be a valid polygon with open holes")
    return plate


def add_mesh(asset, name, vertices, faces):
    ET.SubElement(asset, "mesh", name=name, vertex=text(vertices), face=text(faces))


def build_model(clearance_mm=.5):
    if not 0 <= clearance_mm <= 5:
        raise ValueError("Round peg diameter clearance must be between 0 and 5 mm")
    root = ET.parse(ASSETS / "ffw_sg2.xml").getroot()
    root.set("model", "Cyclo MuJoCo NIST FFW-SG2")
    root.find("compiler").set("meshdir", str(ASSETS / "robot"))
    option = root.find("option")
    option.attrib.update(timestep=".002", gravity="0 0 -9.81", iterations="80",
                         tolerance="1e-9", cone="elliptic", integrator="implicitfast",
                         noslip_iterations="4", noslip_tolerance="1e-8")
    # Position servos for millimetre Cartesian teleoperation; keep the original
    # actuator torque limits and critically damped targets.
    for name, gain in (("DY_80", "2000"), ("DY_70", "2000"), ("DP_42", "700")):
        root.find(f".//default[@class='{name}']/position").set("kp", gain)
    ET.SubElement(root, "size", memory="64M")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1440")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="2")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".45 .45 .45")
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1=".35 .42 .52", rgb2=".12 .14 .18", width="256", height="1536")
    # Fix the mobile base as in the existing manipulation environment.
    for body in world.iter("body"):
        for joint in list(body.findall("freejoint")):
            body.remove(joint)
        body.set("gravcomp", "1")
    base = world.find("body")
    base.set("pos", "-.0254 0 0")
    for index, geom in enumerate(world.iter("geom")):
        geom.set("name", f"robot_geom_{index}")
        if geom.get("class") != "visual":
            geom.set("contype", "1")
            geom.set("conaffinity", "7")
            geom.set("solref", ".006 1")
            geom.set("solimp", ".95 .99 .001")
            geom.set("margin", ".0001")
    for side in ("l", "r"):
        wrist = root.find(f".//body[@name='arm_{side}_link7']")
        ET.SubElement(wrist, "site", name=f"tcp_{side}", pos="0 0 -.2", size=".002", rgba="0 0 0 0")
        # Fixed eye-in-hand extrinsics; calibrated once at reset, never tracking.
        ET.SubElement(wrist, "camera", name="left_wrist" if side == "l" else "right_wrist",
                      pos=".107944 0 -.065588", fovy="45" if side == "l" else "55")
    ET.SubElement(root.find(".//body[@name='head_link2']"), "camera", name="head",
                  pos=".06 0 .035", fovy="65")
    ET.SubElement(world, "camera", name="spectator", pos="1.02 -.91 1.22",
                  quat=text(quat(camera_rotation([1.02, -.91, 1.22], [.40, -.02, .80]))), fovy="32")
    ET.SubElement(world, "light", pos=".2 -.5 2.5", dir="0 0 -1", directional="true", diffuse=".55 .55 .55")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .05",
                  rgba=".17 .19 .22 1", contype="4", conaffinity="3")
    ET.SubElement(world, "geom", name="table", type="box", pos=".65 0 .702",
                  size=".40 .65 .025", rgba=".38 .39 .41 1", contype="4", conaffinity="3",
                  friction=".7 .005 .0001")
    board = ET.SubElement(world, "body", name="nist_board", pos=text(BOARD))
    polygon = plate_polygon(clearance_mm)
    vertices, faces = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
    thickness = LAYOUT["board"]["thickness"]
    plate = trimesh.creation.extrude_triangulation(vertices, faces, thickness)
    plate.vertices[:, 2] -= thickness
    add_mesh(asset, "board_visual", plate.vertices, plate.faces)
    ET.SubElement(board, "geom", name="board_visual", type="mesh", mesh="board_visual",
                  contype="0", conaffinity="0", group="2", rgba=".91 .91 .86 1")
    # Each triangular prism is convex; their union retains every through-hole.
    prism_faces = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                   [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    for index, face in enumerate(faces):
        xy = vertices[face]
        points = np.vstack((np.c_[xy, np.full(3, -thickness)], np.c_[xy, np.zeros(3)]))
        name = f"board_collision_{index}"
        add_mesh(asset, name, points, prism_faces)
        ET.SubElement(board, "geom", name=name, type="mesh", mesh=name, group="3",
                      contype="4", conaffinity="3", solref=".004 1", solimp=".95 .99 .001",
                      friction=".35 .005 .0001", margin="0")
    for x in (-.175, .175):
        for y in (-.175, .175):
            ET.SubElement(board, "geom", type="cylinder", size=".005 .01",
                          pos=text([x, y, -thickness-.01]), rgba=".55 .55 .57 1",
                          contype="4", conaffinity="3")
    for name, part in LAYOUT["parts"].items():
        mesh = trimesh.load(ASSETS / "nist" / part["stl"], force="mesh")
        mesh.apply_scale(.001)
        # The existing layout has already expressed square-12 / hardware yaw in
        # its normalized dimensions. Match those dimensions before centering.
        yaw = 90 if name in ("peg_square_12", "gear_plate", "usb_female", "waterproof_female") else 0
        if yaw:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw), [0, 0, 1]))
        lo, hi = mesh.bounds
        mesh.apply_translation([-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2]])
        add_mesh(asset, name+"_mesh", mesh.vertices, mesh.faces)
        pose = part["pose_kit"] if part["dynamic"] else part["pose"]
        body = ET.SubElement(world, "body", name=name, pos=text(BOARD + np.array(pose[:3])))
        if name.startswith("peg_"):
            ET.SubElement(body, "freejoint", name=name+"_joint")
            ET.SubElement(body, "geom", name=name+"_collision",
                          type="cylinder" if "round" in name else "box",
                          size=text([part["size"][0]/2, part["size"][2]/2] if "round" in name else np.array(part["size"])/2),
                          pos=text([0, 0, part["size"][2]/2]), mass=str(part["mass"]),
                          group="3", contype="2", conaffinity="5", condim="4",
                          friction="1.2 .01 .0002", solref=".004 1", solimp=".95 .99 .001", margin="0")
        ET.SubElement(body, "geom", name=name+"_visual", type="mesh", mesh=name+"_mesh",
                      contype="0" if part["dynamic"] else "4", conaffinity="0" if part["dynamic"] else "3",
                      group="2", rgba=".63 .65 .68 1" if name.startswith(("peg_", "bolt", "nut")) else ".24 .26 .29 1")
    return ET.tostring(root, encoding="unicode"), len(faces)
