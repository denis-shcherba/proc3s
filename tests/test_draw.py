import robotic as ry
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import vtamp.environments.bridge.manipulation as manip
import rowan 
from utils import isolate_red_shapes_from_rgb

SAVE_IMAGE = True
WHITEBOARD_OFFSET = 0.01  # base offset of whiteboard in cms to table
WHITEBOARD_TILT = 10  # tilt of the whiteboard in degrees

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
C.getFrame('l_panda_finger_joint1').setJointState([.01])
C.getFrame("table").setColor([1, 1, 1])


WHITEBOARD_TILT = np.deg2rad(WHITEBOARD_TILT)  # Convert to radians
quaternion_wb = [np.cos(WHITEBOARD_TILT / 2), np.sin(WHITEBOARD_TILT / 2), 0, 0]
table_height = C.getFrame("table").getSize()[2] / 2
whiteboard_thickness = 0.005 
whiteboard_height_adjustment = (1 / np.sqrt(2)) * np.sin(WHITEBOARD_TILT) / 2

C.addFrame("whiteboard", "table")\
    .setColor([1, 1, 1])\
    .setShape(ry.ST.box, [1, 1 / np.sqrt(2), whiteboard_thickness])\
    .setRelativePosition([0, .3, table_height + whiteboard_thickness / 2 + whiteboard_height_adjustment + WHITEBOARD_OFFSET])\
    .setQuaternion(quaternion_wb)

# Pen with whom the robot will draw
C.addFrame("pen", "l_gripper").setColor([1, 0, 0]).setShape(ry.ST.cylinder, [.1, .01]).setRelativePosition([0, 0, -.05])

homeState = C.getJointState()

lines = 0


def draw_line(x0, y0, z0, x1, y1, z1):
    """
    Draws a line by moving the robots gripper between two points and 
    placing visual markers if the pen is in contact with the whiteboard.

    Args:
        x0, y0, z0 (float): Start coordinates.
        x1, y1, z1 (float): End coordinates.
    """
    global lines

    for i in range(2):
        if i == 1:
            C.addFrame("tmp").setPosition(C.getFrame("l_gripper").getPosition())

        man = manip.ManipulationModelling()
        man.setup_inverse_kinematics(C, accumulated_collisions=False)

        target = [x0, y0, z0] if i == 0 else [x1, y1, z1]
        man.komo.addObjective([1], ry.FS.position, ["l_gripper"], ry.OT.eq, 1, target)

        for _ in range(2):
            man.komo.addObjective([], ry.FS.vectorZ, ["l_gripper"], ry.OT.eq, [1], [0, 0, 1])

        ret = man.solve()
        print('    IK:', ret)

        feasible = man.feasible
        path = man.path

        if not feasible:
            print('  -- infeasible')
            continue

        man = manip.ManipulationModelling()
        man.setup_point_to_point_motion(C, path[0], accumulated_collisions=False)

        if i == 1:
            delta = np.array(target) - C.getFrame("l_gripper").getPosition()
            delta /= np.linalg.norm(delta)
            projection_matrix = np.eye(3) - np.outer(delta, delta)
            man.komo.addObjective([], ry.FS.positionDiff, ['l_gripper', "tmp"], ry.OT.eq, 100 * projection_matrix)

        ret = man.solve()
        path = man.path
        feasible = feasible and man.feasible

        if not feasible:
            print('  -- infeasible')
            continue



        for t in range(path.shape[0]):
            neg_dist = C.eval(ry.FS.negDistance, ["pen", "whiteboard"])[0]
            if neg_dist > 0:
                pen_pos = C.getFrame("pen").getPosition()

                # Get the whiteboard's pose (position and quaternion)
                whiteboard_pose = C.getFrame("whiteboard").getPose()
                whiteboard_pos = whiteboard_pose[:3]
                whiteboard_quat = whiteboard_pose[3:]

                # Compute the whiteboard's rotation matrix using rowan
                R = rowan.to_matrix(whiteboard_quat)

                # Transform the pen position into the whiteboard's local coordinate system
                pen_pos_local = np.dot(R.T, pen_pos - whiteboard_pos)

                # Project the pen position onto the whiteboard's plane (z = 0 in local coordinates)
                pen_pos_local[2] = 0

                # Transform the projected position back to the global coordinate system
                pen_pos_projected_global = np.dot(R, pen_pos_local) + whiteboard_pos

                # Add the sphere at the projected position
                C.addFrame(f"circle_{lines}_{i}_{t}")\
                    .setShape(ry.ST.sphere, size=[.01])\
                    .setPosition(pen_pos_projected_global)\
                    .setColor([1, 0, 0])


            C.setJointState(path[t])
            C.view(False)
            time.sleep(0.05)

        lines += 1

draw_line(.2, .4, .76, .2, .35, .76)
C.setJointState(homeState)
draw_line(.0, .4, .76, .0, .35, .76)
C.setJointState(homeState)
draw_line(.1, .3, .76, .1, .275, .76)
C.setJointState(homeState)
draw_line(.2, .2, .76, .1, .1, .76)
C.setJointState(homeState)
draw_line(.1, .1, .76, .0, .2, .76)

C.view(True)
to_stay = ["world", "table", "whiteboard", "cameraTop"]

C.getFrame("cameraTop").setQuaternion([0, 1, 0, 0]).setPosition([0, .3, 1.5])
C.view(True)
for frame in C.getFrameNames():
    if frame not in to_stay and "circle_" not in frame:
        C.delFrame(frame)
        
CameraView = ry.CameraView(C)
CameraView.setCamera(C.getFrame("cameraTop"))
image, _ = CameraView.computeImageAndDepth(C)
plt.imshow(image)
plt.show()

image_without_background = isolate_red_shapes_from_rgb(image, background_color=(255, 255, 255))
plt.imshow(image_without_background)

# Save the image without background
if SAVE_IMAGE:
    plt.axis('off')  # Optional: Remove axes for clean output

    # Get the next available filename
    i = 0
    while os.path.exists(f"output_{i}.png"):
        i += 1
    filename = f"output_{i}.png"

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved image as {filename}")


plt.show()

