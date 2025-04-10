from __future__ import annotations

import copy
import time
import rowan
import logging
import numpy as np
import robotic as ry
from typing import List, Tuple

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
    

def pick_place_manipulation(C: ry.Config,
                            frame_name: str,
                            pick_dir: str,
                            place_dir: str,
                            pos: Tuple[float],
                            yaw: float,
                            compute_collisions: bool=True) -> manip.ManipulationModelling:
    x, y, z = pos
    M = manip.ManipulationModelling()
    M.setup_pick_and_place_waypoints(C, "l_gripper", frame_name, accumulated_collisions=compute_collisions)
    
    M.grasp_box(1., "l_gripper", frame_name, "l_palm", pick_dir)

    if z == None:
        M.place_box(2., frame_name, "table", "l_palm", place_dir)
        M.target_relative_xy_position(2., frame_name, "table", [x, y])
    else:
        table_frame = C.getFrame("table")
        table_offset = table_frame.getPosition()[2] + table_frame.getSize()[2]*.5
        if z < table_offset:
            z += table_offset
        M.place_box(2., frame_name, "table", "l_palm", place_dir, on_table=False)
        M.target_position(2., frame_name, [x, y, z])

    if yaw != None:
        
        if place_dir == "x" or place_dir == "xNeg":
            feature = ry.FS.vectorY
        
        elif place_dir == "y" or place_dir == "yNeg":
            feature = ry.FS.vectorX
        
        elif place_dir == "z" or place_dir == "zNeg":
            feature = ry.FS.vectorY
        
        else:
            raise Exception(f"'{place_dir}' is not a valid up vector for a place motion!")

        yaw += np.pi*.5
        target = np.array([np.cos(yaw), -np.sin(yaw), .0])
        if "Neg" in place_dir: target *= -1
        M.komo.addObjective([2.], feature, [frame_name], ry.OT.eq, [1e1], target)
    
    return M



class BridgeEnv(Environment):
    def __init__(self, task: Task, **kwargs):

        super().__init__(task)

        self.compute_collisions = True
        
        self.base_config: ry.Config = self.task.setup_env()
        self.base_config.view(False, "Base Config")
        self.C: ry.Config = self.task.setup_env()
        self.C.view(False, "Working Config")
        self.initial_state = self.reset()
        self.qHome = self.C.getJointState()

    def step(self, action: Action, vis: bool=True):
        info = {"constraint_violations": []}

        if not self.feasible:
            self.C.view()
            self.t = self.t + 1
            return self.state, False, 0, info
        
        self.feasible = False

        if action.name == "pick":
            assert self.to_be_picked == None
            self.to_be_picked = action.params
            self.feasible = True
        
        elif action.name == "place_sr":
            assert self.to_be_picked != None

            frame_name = self.to_be_picked[0]
            pick_dir = self.to_be_picked[1]
            x, y, z = action.params[:3]
            rotated, yaw = action.params[3:5]
            
            grasp_dirs = ["x", "y"] if pick_dir == None else [pick_dir]
            for grasp_dir in grasp_dirs:
                if rotated and grasp_dir == 'x':
                    place_dirs = ['y', 'yNeg']
                elif rotated and grasp_dir == 'y':
                    place_dirs = ['x', 'xNeg']
                elif not rotated:
                    place_dirs = ['z', 'zNeg']


                for place_dir in place_dirs:
                    M = pick_place_manipulation(self.C,
                                                frame_name,
                                                grasp_dir,
                                                place_dir,
                                                (x, y, z),
                                                yaw,
                                                self.compute_collisions)

                    M.solve(verbose=0)
                    if M.feasible:

                        M1 = M.sub_motion(0, accumulated_collisions=self.compute_collisions)
                        # M1.keep_distance([.3,.7], "l_palm", frame_name, margin=.05)
                        # M1.retract([.0, .2], "l_gripper")
                        # M1.approach([.8, 1.], "l_gripper")
                        path1 = M1.solve(verbose=0)

                        M2 = M.sub_motion(1, accumulated_collisions=self.compute_collisions)
                        # M2.keep_distance([.2, .8], "table", frame_name, .04)
                        # M2.keep_distance([], "l_palm", frame_name)
                        path2 = M2.solve(verbose=0)
                        
                        if M1.feasible and M2.feasible:
                            
                            if vis:
                                for q in path1:
                                    self.C.setJointState(q)
                                    self.C.view()
                                    time.sleep(.1)
                                self.C.attach("l_gripper", frame_name)
                                
                                for q in path2:
                                    self.C.setJointState(q)
                                    self.C.view()
                                    time.sleep(.1)
                                self.C.attach("table", frame_name)
                            
                            else:
                                self.C.setJointState(path1[-1])
                                self.C.attach("l_gripper", frame_name)
                                self.C.view()
                                self.C.setJointState(path2[-1])
                                self.C.attach("table", frame_name)
                                self.C.view()
                                time.sleep(.05) # This is to kind of see that the last block in Bridge building also gets placed.

                            self.feasible = True
                            self.to_be_picked = None
                            break

                if self.feasible:
                    break
        
        else:
            raise NotImplementedError
        
        if not self.feasible:
            info["constraint_violations"].append("idk")

        self.C.view()
        self.t = self.t + 1
        self.state = self.getState()
        return self.state, False, 0, info
    
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
        self.to_be_picked: List[str] = None

        return self.state
    
    def getState(self):

        state = BridgeState()
        state.frames = []
        
        relevant_frames = ["block_red", "block_green", "block_blue"]
        for f in relevant_frames:
        
            C_frame = self.C.getFrame(f)
        
            pos = C_frame.getPosition()
            size = C_frame.getSize()
            rot = rowan.to_euler(C_frame.getQuaternion(), convention="xyz") # Rotations need further testing
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