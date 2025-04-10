from __future__ import annotations

import numpy as np
import robotic as ry

from vtamp.environments.bridge.env import BridgeEnv, Task


def create_config(vertical_blocks: bool=True) -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

    C.delFrame("panda_collCameraWrist")
    C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

    names = ["red", "green", "blue"]

    # Objects
    for i in range(3):
        color = [0., 0., 0.]
        color[i%3] = 1.
        size_xyz = [.04, .04, .12] if vertical_blocks else [.04, .12, .04]
        C.addFrame(f"block_{names[i]}") \
            .setPosition([(i%3)*.15, (i//3)*.1+.1, .71]) \
            .setShape(ry.ST.ssBox, size=[*size_xyz, 0.005]) \
            .setColor(color) \
            .setContact(1) \
            .setMass(.1)
    
    return C


class BuildPlanarTriangle(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_config()

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
        
            return total_cost[0]
    

class TestTask(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def setup_env(self):
        return create_config()

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
        return create_config(vertical_blocks=False)
    
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
    