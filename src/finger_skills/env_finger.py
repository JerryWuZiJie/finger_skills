'''
This run on real robot
'''
import os
import re
import time
import sys
import threading

import numpy as np
import matplotlib.pylab as plt
import pinocchio as pin
import pybullet

from bullet_utils.env import BulletEnv
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from robot_properties_nyu_finger.wrapper import NYUFingerRobot

from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController

# get absolute path for finger_skills
PARENT_FOLDER = 'finger_skills'
current_path = os.getcwd()
result = current_path.find(PARENT_FOLDER)
if result == -1:
    print('check directory path!!!')
    sys.exit(0)
current_path = current_path[:result+len(PARENT_FOLDER)]

# setup some constants
ID0 = 'finger0_lower_to_tip_joint'  # finger0 end effector id
ID1 = 'finger1_lower_to_tip_joint'  # finger1 end effector id


def cal_forwardK(pin_robot, id_ee):
    '''
    calculate position matrix (transition and rotation)
    '''
    # get frame id
    FRAME_ID = pin_robot.model.getFrameId(id_ee)

    # get pose
    return pin.updateFramePlacement(pin_robot.model, pin_robot.data, FRAME_ID)


def cal_oriented_j(pin_robot, id_ee, q):
    '''
    calculate oriented jacobian of the end effector
    '''
    # get frame id
    FRAME_ID = pin_robot.model.getFrameId(id_ee)

    # get pose
    pose = pin.updateFramePlacement(pin_robot.model, pin_robot.data, FRAME_ID)

    # get oriented jacobian
    body_jocobian = pin.computeFrameJacobian(
        pin_robot.model, pin_robot.data, q, FRAME_ID)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = pose.rotation
    Ad[3:, 3:] = pose.rotation

    return Ad @ body_jocobian


# controllers for running ----------------------
class Controller:
    def __init__(self, head, id_ee, dt=0.001):
        # setup head
        self.head = head
        # pos and vel is reference, will update automatically
        self.joint_positions = self.head.get_sensor('joint_positions')
        self.joint_velocities = self.head.get_sensor('joint_velocities')

        # setup pinocchio robot model
        self.id = id_ee
        if '0' in self.id:
            self.robot = NYUFingerDoubleConfig0.buildRobotWrapper()
        else:
            self.robot = NYUFingerDoubleConfig1.buildRobotWrapper()

        # setup step and robot
        self.dt = dt
        self.current_step = 0
        self.robot.framesForwardKinematics(self.joint_positions)

    def set_target(self, des_pos, des_vel=None):
        # set des_pos
        if type(des_pos) == float or type(des_pos) == int:
            des_pos = [des_pos] * 3
        self.des_pos = np.array(des_pos)

        # set des_vel
        if des_vel is None:
            des_vel = np.zeros(3)
        elif type(des_vel) == float or type(des_vel) == int:
            des_vel = [des_vel] * 3
        self.des_vel = np.array(des_vel)

    def warmup(self, thread):  # TODO:  reset
        self.robot.framesForwardKinematics(self.joint_positions)
        self.current_step = 0

    def run(self, thread):  # TODO: step
        # update robot kinematics
        self.robot.framesForwardKinematics(self.joint_positions)
        self.current_step += 1

        # # calculate position and oriented jacobian
        # pose_ee = cal_forwardK(self.robot, self.id).translation
        # oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

        # send torque
        self.tau = np.zeros(self.robot.nq)
        self.head.set_control('ctrl_joint_torques', self.tau)

        # return obs, rew, done, info


class EnvFingers:
    def __init__(self, des_pos, ee0_id=ID0, ee1_id=ID1):
        # desired position of the box
        self.des_pose = np.array(des_pos)
        # threshold for done condition
        self.threshold = 0.01 ** 2  # meter, it is the square of distance
        # set initial box position
        self.box_default_pos = [0., -0.2, 0.05]
        self.box_default_ori = (0., 0., 0., 1.)
        # end effector id
        self.ee0_id = ee0_id
        self.ee1_id = ee1_id

        # init BulletEnv
        self.bullet_env = BulletEnv()  # BulletEnvWithGround()

        # set initial view point
        pybullet.resetDebugVisualizerCamera(.7, 0, -15, (0., 0., 0.2))
        # Disable the gui controller as we don't use them.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        # Create a robot instance
        config0 = NYUFingerDoubleConfig0()
        config1 = NYUFingerDoubleConfig1()
        self.finger0 = NYUFingerRobot(config=config0)
        self.finger1 = NYUFingerRobot(config=config1)
        self.bullet_env.add_robot(self.finger0)
        self.bullet_env.add_robot(self.finger1)

        # add a plane
        self.planeid = pybullet.loadURDF(os.path.join(
            current_path, "src/urdf/plane.urdf"))
        pybullet.resetBasePositionAndOrientation(
            self.planeid, [0., 0., 0.], (0., 0., 0., 1.))
        pybullet.changeDynamics(
            self.planeid, -1, lateralFriction=5., rollingFriction=0)

        # add a box
        self.boxid = pybullet.loadURDF(os.path.join(
            current_path, "src/urdf/box.urdf"))
        pybullet.resetBasePositionAndOrientation(
            self.boxid, self.box_default_pos, self.box_default_ori)
        pybullet.changeDynamics(
            self.boxid, -1, lateralFriction=0.5, spinningFriction=0.5)

        # contruct name to id map
        self.bullet_joint_map = {}
        for ji in range(pybullet.getNumJoints(self.finger0.robotId)):
            self.bullet_joint_map[pybullet.getJointInfo(self.finger0.robotId, ji)[1].decode("UTF-8")] = ji

    def reset(self):
        # set the robot initial state in theta
        self.finger0.reset_state(
            np.zeros(self.finger0.nq), np.zeros(self.finger0.nv))
        self.finger1.reset_state(
            np.zeros(self.finger1.nq), np.zeros(self.finger1.nv))
        
        # reset box position
        pybullet.resetBasePositionAndOrientation(
            self.boxid, self.box_default_pos, self.box_default_ori)

        observation = np.array([*self.box_default_pos])  # TODO
        return observation

    def step(self, action):
        # update kinematic
        q0, dq0 = self.finger0.get_state()
        self.finger1.pin_robot.forwardKinematics(q0)
        q1, dq1 = self.finger1.get_state()
        self.finger1.pin_robot.forwardKinematics(q1)
        # finger tip position
        ee0_pose = cal_forwardK(self.finger0.pin_robot, self.ee0_id).translation
        ee1_pose = cal_forwardK(self.finger1.pin_robot, self.ee1_id).translation
        ee0_check = pybullet.getJointState(self.finger0.robotId, self.bullet_joint_map[self.ee0_id])
        # print(ee0_pose, ee0_check)
        # box position
        box_pose = pybullet.getBasePositionAndOrientation(self.boxid)[0]

        # finger0 control -------------------------------------------------------
        joint_torques0 = np.random.normal(scale=0.05, size=self.finger0.nq)
        self.finger0.send_joint_command(joint_torques0)
        # -----------------------------------------------------------------------

        # finger1 control -------------------------------------------------------
        joint_torques1 = np.random.normal(scale=0.05, size=self.finger1.nq)
        self.finger1.send_joint_command(joint_torques1)
        # -----------------------------------------------------------------------

        # we send them to the robot and do one simulation step
        self.bullet_env.step(sleep=True)

        # observation compose of finger tip position and box position
        observation = [*ee0_pose, *ee1_pose, *box_pose]

        # reward is the difference between box and desired location
        reward = -sum((box_pose-self.des_pose)**2)

        # done if box is close to desired location
        if -reward < self.threshold:
            done = True
        else:
            done = False

        # info is None for now
        info = None

        # TODO
        return observation, reward, done, info

    def render(self):
        # TODO
        pass

    def close(self):
        pybullet.disconnect()


if __name__ == '__main__':

    env = EnvFingers(des_pos=[1, 1, 1])

    for i in range(1):
        env.reset()
        for i in range(1000):
            obs, rew, done, info = env.step(None)
        print(obs)

    env.close()
