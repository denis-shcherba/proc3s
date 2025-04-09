from __future__ import annotations

import copy
import time
import rowan
import logging
import numpy as np
import robotic as ry
from typing import List

from shapely.geometry import Point
from vtamp.environments.utils import Action, Environment, State, Task
import vtamp.environments.bridge.manipulation as manip
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
# Used as imports for the LLM-generated code
__all__ = ["Frame", "BridgeState"]


@dataclass
class Frame:
    name: str
    x_pos: float
    y_pos: float
    z_pos: float
    x_size: float
    y_size: float
    z_size: float
    x_rot: float
    y_rot: float
    z_rot: float
    color: List[float]

    def __str__(self):
        return 'Frame(name="{}", x_pos={}, y_pos={}, z_pos={}, x_size={}, y_size={}, z_size={}, x_rot={}, y_rot={}, z_rot={}, color="{}")'.format(
            self.name,
            round(self.x_pos, 2),
            round(self.y_pos, 2),
            round(self.z_pos, 2),
            round(self.x_size, 2),
            round(self.y_size, 2),
            round(self.z_size, 2),
            round(self.x_rot, 2),
            round(self.y_rot, 2),
            round(self.z_rot, 2),
            [round(self.color[0]), round(self.color[1]), round(self.color[2])],
        )


@dataclass
class BridgeState(State):
    frames: List[Frame] = field(default_factory=list)

    def __str__(self):
        return "BridgeState(frames=[{}])".format(
            ", ".join([str(o) for o in self.frames])
        )

    def getFrame(self, name: str) -> Frame:
        for f in self.frames:
            if f.name == name:
                return f
        return None


def create_config() -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

    C.delFrame("panda_collCameraWrist")
    C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

    names = ["red", "green", "blue"]

    # Objects
    for i in range(3):
        color = [0., 0., 0.]
        color[i%3] = 1.
        C.addFrame(f"block_{names[i]}") \
            .setPosition([(i%3)*.15, (i//3)*.1+.1, .71]) \
            .setShape(ry.ST.ssBox, size=[.04, .04, .12, 0.005]) \
            .setColor(color) \
            .setContact(1) \
            .setMass(.1)
    
    return C

def create_horizontal_config() -> ry.Config:
    #TODO more center, more space between blocks, make blocks on table 
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

    C.delFrame("panda_collCameraWrist")
    C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

    names = ["red", "green", "blue"]

    # Objects
    for i in range(3):
        color = [0., 0., 0.]
        color[i%3] = 1.
        C.addFrame(f"block_{names[i]}") \
            .setPosition([(i%3)*.15, (i//3)*.1+.1, .71]) \
            .setShape(ry.ST.ssBox, size=[.04, .12, .04, 0.005]) \
            .setColor(color) \
            .setContact(1) \
            .setMass(.1)
    
    return C

class BuildPlanarTriangle(Task):
    #TODO prolly
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_horizontal_config()

    def get_reward(self, env: BridgeEnv):
        return 0
    
    def get_cost(self, env: BridgeEnv):

            red_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_red", "block_green"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_red", "block_blue"])[0][0])
            green_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_green", "block_blue"])[0][0])
            blue_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_blue", "block_red"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_blue", "block_green"])[0][0])

            red_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_red", "block_green"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_red", "block_blue"])[0][1])
            green_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_green", "block_blue"])[0][1])
            blue_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_blue", "block_red"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_blue", "block_green"])[0][1])

            # Distance of one cm between triangle sides
            red_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_red", "block_green"])[0]+.01)**2
            green_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_green", "block_blue"])[0]+.01)**2
            blue_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_blue", "block_green"])[0]+.01)**2

            total_cost = red_block_error + green_block_error + blue_block_error
        
            return total_cost
    

class TestTask(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_horizontal_config()

    def get_reward(self, env: BridgeEnv):
        return 0
    
    def get_cost(self, env: BridgeEnv):

        red_block_error = 30*(env.C.eval(ry.FS.negDistance, ["block_red", "block_green"])[0]+.04)**2
        green_block_error = 30*(env.C.eval(ry.FS.negDistance, ["block_green", "block_blue"])[0]+.04)**2

        blue_block_error = 10*(env.C.eval(ry.FS.positionDiff, ["block_blue", "block_green"])[0][1])**2
        green_block_error += 10*(env.C.eval(ry.FS.positionDiff, ["block_green", "block_red"])[0][1])**2

        total_cost = red_block_error + green_block_error + blue_block_error
        return total_cost[0]

class BuildPlanarI(Task):
    # TODO look into CMA-Es issue, doesnt output the best solution but the last
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_horizontal_config()
    
    def get_reward(self, env: BridgeEnv):
        return 0
    
    def get_cost(self, env: BridgeEnv):
        red_block_error = np.abs(env.C.eval(ry.FS.scalarProductXX, ["block_red", "block_green"])[0][0])
        green_block_error = np.abs(env.C.eval(ry.FS.scalarProductXX, ["block_green", "block_blue"])[0][0])
        blue_block_error = np.abs(env.C.eval(ry.FS.scalarProductXY, ["block_blue", "block_red"])[0][0])
        
        # alignment things
        green_block_error += 10 * (np.abs(np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][0])-.08)+np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][1]))

        total_cost = red_block_error + green_block_error + blue_block_error

        if total_cost<.01:
            env.C.view(True)


        return total_cost
    


class BuildBridge(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_config()

    def get_reward(self, env: BridgeEnv):
        return 0
    
    def get_cost(self, env: BridgeEnv):

        red_block = env.C.getFrame("block_red")
        green_block = env.C.getFrame("block_green")
        blue_block = env.C.getFrame("block_blue")

        red_block_error = 0
        green_block_error = 0
        blue_block_error = 0

        # Positions
        green_block_error += np.abs(np.linalg.norm(green_block.getPosition() - red_block.getPosition()) - 0.12)
        blue_block_error += np.abs((blue_block.getPosition()[2] - red_block.getPosition()[2]) - .06 - .02)

        # Rotations
        blue_block_error += np.abs(env.C.eval(ry.FS.scalarProductZZ, ["block_blue", "table"])[0][0])
        total_cost = red_block_error + green_block_error + blue_block_error
        
        return total_cost
    

class PlaceRed(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_config()

    def get_reward(self, env: BridgeEnv):
        return 0
    
    def get_cost(self, env: BridgeEnv):
        red_block = env.C.getFrame("block_red")
        total_cost = np.linalg.norm(red_block.getPosition()[:2] - np.array([.3, .3]))**2
        return total_cost


class BridgeEnv(Environment):
    def __init__(self, task: Task, **kwargs):

        super().__init__(task)

        self.compute_collisions = False
        
        self.base_config: ry.Config = self.task.setup_env()
        self.base_config.view(False, "Base Config")
        self.C: ry.Config = self.task.setup_env()
        self.C.view(False, "Working Config")
        self.initial_state = self.reset()

    def step(self, action: Action, vis: bool=True):
        info = {"constraint_violations": []}

        if not self.feasible:
            self.C.view()
            self.t = self.t + 1
            return self.state, False, 0, info
        
        self.feasible = False

        if action.name == "pick":

            assert self.grabbed_frame == ""

            frame = action.params[0]
            pick_axis = action.params[1]
            
            if pick_axis == None:
                graspDirections = ['x', 'y']
            else:
                graspDirections = [pick_axis]
            for gd in graspDirections:

                M = manip.ManipulationModelling()
                M.setup_sequence(self.C, 1, accumulated_collisions=self.compute_collisions)
                M.grasp_box(1., "l_gripper", frame, "l_palm", gd)
                M.solve(verbose=0)
                
                if M.feasible:
                    M1 = M.sub_motion(0, accumulated_collisions=self.compute_collisions)
                    M1.no_collisions([.3,.7], ["l_palm", frame], margin=.05)
                    M1.retract([.0, .2], "l_gripper")
                    M1.approach([.8, 1.], "l_gripper")
                    self.path = M1.solve(verbose=0)
                
                    if M1.feasible:
                        if vis:
                            for q in self.path:
                                self.C.setJointState(q)
                                self.C.view()
                                time.sleep(.1)
                        else:
                            self.C.setJointState(self.path[-1])
                        self.C.attach("l_gripper", frame)

                        self.grabbed_frame = frame
                        self.grasp_direction = gd
                        self.feasible = True
                        break
        
        elif action.name == "place":
            
            assert self.grabbed_frame != ""

            x = action.params[0]
            y = action.params[1]
            z = action.params[2]
            roll = action.params[3]
            pitch = action.params[4]
            yaw = action.params[5]

            if roll == None and pitch == None and yaw == None:
                place_direction = 'z'
            else:
                place_direction = None

            self.feasible = False

            M = manip.ManipulationModelling()
            M.setup_sequence(self.C, 1, accumulated_collisions=self.compute_collisions, joint_limits=False, homing_scale=.1)
            
            if z == None:
                M.place_box(1., self.grabbed_frame, "table", "l_palm", place_direction)
                M.target_relative_xy_position(1., self.grabbed_frame, "table", [x, y])
            else:
                table_frame = self.C.getFrame("table")
                table_offset = table_frame.getPosition()[2] + table_frame.getSize()[2]*.5
                if z < table_offset:
                    z += table_offset
                M.place_box(1., self.grabbed_frame, "table", "l_palm", place_direction, on_table=False)
                M.target_position(1., self.grabbed_frame, [x, y, z])

            if roll != None:
                M.komo.addObjective([.8, 1.], ry.FS.scalarProductYY, ["table", self.grabbed_frame], ry.OT.eq, [1e1], [np.cos(roll)])
                if pitch == None and yaw == None:
                    M.komo.addObjective([.8, 1.], ry.FS.vectorX, [self.grabbed_frame], ry.OT.eq, [1e1], [1., 0., 0.])
            if pitch != None:
                M.komo.addObjective([.8, 1.], ry.FS.scalarProductZZ, ["table", self.grabbed_frame], ry.OT.eq, [1e1], [np.cos(pitch)])
                if roll == None and yaw == None:
                    M.komo.addObjective([.8, 1.], ry.FS.vectorY, [self.grabbed_frame], ry.OT.eq, [1e1], [0., 1., 0.])
            if yaw != None:
                M.komo.addObjective([.8, 1.], ry.FS.scalarProductXX, ["table", self.grabbed_frame], ry.OT.eq, [1e1], [np.cos(yaw)])
                if roll == None and pitch == None:
                    M.komo.addObjective([.8, 1.], ry.FS.vectorZ, [self.grabbed_frame], ry.OT.eq, [1e1], [0., 0., 1.])

            M.solve(verbose=0)
            if M.feasible:

                M1 = M.sub_motion(0, accumulated_collisions=self.compute_collisions)
                self.path = M1.solve(verbose=0)
                if M1.feasible:
                    if vis:
                        for q in self.path:
                            self.C.setJointState(q)
                            self.C.view()
                            time.sleep(.1)
                    else:
                        self.C.setJointState(self.path[-1])
                    self.C.attach("table", self.grabbed_frame)

                    self.grabbed_frame = ""
                    self.grasp_direction = ""
                    self.feasible = True
        
        elif action.name == "place_sr":
            assert self.grabbed_frame != ""

            x = action.params[0]
            y = action.params[1]
            z = action.params[2]
            rotated = action.params[3]
            yaw = action.params[4]


            if rotated and self.grasp_direction == 'x':
                place_direction = ['y', 'yNeg']
            elif rotated and self.grasp_direction == 'y':
                place_direction = ['x', 'xNeg']
            elif not rotated:
                place_direction = ['z', 'zNeg']

            self.feasible = False

            Ms = []
            for i, direction in enumerate(place_direction):
                for j in range(2 if yaw is not None else 1):
                    M = manip.ManipulationModelling()
                    M.setup_sequence(self.C, 1, accumulated_collisions=self.compute_collisions, joint_limits=False, homing_scale=.1)

                    if z == None:
                        M.place_box(1., self.grabbed_frame, "table", "l_palm", direction)
                        M.target_relative_xy_position(1., self.grabbed_frame, "table", [x, y])
                    else:
                        table_frame = self.C.getFrame("table")
                        table_offset = table_frame.getPosition()[2] + table_frame.getSize()[2]*.5
                        if z < table_offset:
                            z += table_offset
                        M.place_box(1., self.grabbed_frame, "table", "l_palm", direction, on_table=False)
                        M.target_position(1., self.grabbed_frame, [x, y, z])

                    if yaw != None:
                        if direction == "x" or direction == "xNeg":
                            feature = ry.FS.scalarProductXZ
                        elif direction == "y" or direction == "yNeg":
                            feature = ry.FS.scalarProductXX
                        elif direction == "z" or direction == "zNeg":
                            feature = ry.FS.scalarProductXX
                        else:
                            raise Exception(f"'{place_direction}' is not a valid up vector for a place motion!")
                        
                        M.komo.addObjective([.8, 1.], feature, ["table", self.grabbed_frame], ry.OT.eq, [1e1], [np.cos(yaw+j*np.pi)])

                    M.solve(verbose=0)
                    Ms.append((M, M.ret.sos + M.ret.eq))
                
            Ms.sort(key=lambda x: x[1])  # Sort by cost (index 1)
            M = Ms[0][0]

            if M.feasible:

                M1 = M.sub_motion(0, accumulated_collisions=self.compute_collisions)
                self.path = M1.solve(verbose=0)
                if M1.feasible:
                    for q in self.path:
                        self.C.setJointState(q)
                        # self.C.view()
                        # time.sleep(.1)
                    self.C.attach("table", self.grabbed_frame)

                    self.grabbed_frame = ""
                    self.grasp_direction = ""
                    self.feasible = True
        
        else:
            raise NotImplementedError

        self.C.view()
        self.t = self.t + 1

        return self.getState(), False, 0, info
    
    @staticmethod
    def sample_twin(real_env: BridgeEnv, obs, task: Task, **kwargs) -> BridgeEnv:
        twin = BridgeEnv(task)
        twin.C = ry.Config()
        twin.C.addConfigurationCopy(real_env.C)
        twin.state = copy.deepcopy(obs)
        twin.initial_state = copy.deepcopy(obs)
        return twin

    def reset(self):
        relevant_frames = ["block_red", "block_green", "block_blue"]
        for f in relevant_frames:
            self.C.attach("table", f)
        q = self.base_config.getJointState()
        C_state = self.base_config.getFrameState()
        self.C.setJointState(q)
        self.C.setFrameState(C_state)
        self.C.view()
        self.state = self.getState()
        self.t = 0
        self.feasible = True

        self.grabbed_frame = ""
        self.grasp_direction = ""
        
        return self.getState()
    
    def getState(self):

        state = BridgeState()
        state.frames = []
        
        relevant_frames = ["block_red", "block_green", "block_blue"]
        for f in relevant_frames:
        
            C_frame = self.C.getFrame(f)
        
            pos = C_frame.getPosition()
            size = C_frame.getSize()
            rot = rowan.to_euler(C_frame.getQuaternion())
            color = C_frame.getMeshColors()[0][:3]
        
            frame = Frame(f, *pos, *size[:3], *rot, color)
            state.frames.append(frame)
        
        return state

    def render(self):
        self.C.view(True)

    def compute_cost(self):
        self.C.view()
        cost = self.task.get_cost(self)
        return cost