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
SIMULATION = True  # simulation or run on real robot
SHOW_TIMING = False  # show timing log at the end
FINGER0_ONLY = False  # if true, only control finger0
ID0 = 'finger0_lower_to_tip_joint'  # finger0 end effector id
ID1 = 'finger1_lower_to_tip_joint'  # finger1 end effector id
DT = 0.001  # step time


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


def cal_inverseK(pin_robot, id_ee, des_pos, cur_q):
    '''
    given desire position, return desire q
    '''
    # THIS DOESN'T WORK AND USE IMPEDANCE CONTROL INSTEAD!!!

    # get frame id
    FRAME_ID = pin_robot.model.getFrameId(id_ee)
    des_pos = np.array(des_pos)

    # q = np.array(cur_q)  # make a copy of current position
    q = np.array(cur_q)
    print('-' * 50)
    pin_robot.framesForwardKinematics(q)
    pose_ee = pin.updateFramePlacement(
        pin_robot.model, pin_robot.data, FRAME_ID).translation
    print('\tdesire pos:', des_pos)
    print('\tcurrent pos:', pose_ee)
    eps = 1e-4
    IT_MAX = 10000
    DT = 1
    MIN_DT = 1e-3
    damp = 1e-12

    for _ in range(IT_MAX):
        pin_robot.framesForwardKinematics(q)
        pose_ee = pin.updateFramePlacement(
            pin_robot.model, pin_robot.data, FRAME_ID).translation
        err = pose_ee - des_pos
        if np.linalg.norm(err) < eps:
            print("\tConvergence achieved!")
            break
        J = pin_robot.computeFrameJacobian(q, FRAME_ID)[:3]
        v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(3), err))
        q = pin.integrate(pin_robot.model, q, v*DT)
        if DT > MIN_DT:
            DT *= 0.99
    else:
        print("\n\tWarning: the iterative algorithm has not reached convergence to the desired precision\n")

    q = q / 180 * np.pi
    print('\tfinal pos:', pose_ee)
    print('\tresult:', q)
    print('\tfinal error:', err.T)
    print('\tfinal dt:', DT)
    print('-' * 50)
    return q

    '''
    compute interpolated trajectory and return a list with pos and vel
    '''
    # compute coefficient
    a5 = 6/(movement_duration**5)
    a4 = -15/(movement_duration**4)
    a3 = 10/(movement_duration**3)
    diff = th_goal - th_init

    trajectory = []
    t = 0
    for i in range(int(movement_duration/dt)):

        # now we compute s and ds/dt
        s = a3 * t**3 + a4 * t**4 + a5 * t**5
        ds = 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4

        # now we compute th and dth/dt (the angle and its velocity)
        th = th_init + s * diff
        dth = ds * diff

        trajectory.append((th, dth))
        t += dt

    # make sure the last one has desired velocity (mostly zero)
    trajectory[-1] = (trajectory[-1][0], des_vel)

    return trajectory


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
    def __init__(self):
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
        plane = pybullet.loadURDF(os.path.join(
            current_path, "src/urdf/plane.urdf"))
        pybullet.resetBasePositionAndOrientation(
            plane, [0., 0., 0.], (0., 0., 0., 1.))
        pybullet.changeDynamics(
            plane, -1, lateralFriction=5., rollingFriction=0)

        # add a box
        box = pybullet.loadURDF(os.path.join(
            current_path, "src/urdf/box.urdf"))
        pybullet.resetBasePositionAndOrientation(
            box, [0., -0.2, 0.05], (0., 0., 0., 1.))
        pybullet.changeDynamics(
            box, -1, lateralFriction=0.5, spinningFriction=0.5)

    def reset(self):
        # set the robot initial state in theta
        self.finger0.reset_state(np.zeros(self.finger0.nq), np.zeros(self.finger0.nv))
        self.finger1.reset_state(np.zeros(self.finger1.nq), np.zeros(self.finger1.nv))

        observation = None  # TODO
        return observation
    
    def step(self, action):
        q0, dq0 = finger0.get_state()
        q1, dq1 = finger1.get_state()

        # TODO
        pass

    def render(self):
        # TODO
        pass


if __name__ == '__main__':

    # init BulletEnv and setup robot ----------------------
    bullet_env = BulletEnv()  # BulletEnvWithGround()
    # set initial view point
    pybullet.resetDebugVisualizerCamera(.7, 0, -15, (0., 0., 0.2))
    # Disable the gui controller as we don't use them.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    # Create a robot instance. This initializes the simulator as well.
    config0 = NYUFingerDoubleConfig0()
    config1 = NYUFingerDoubleConfig1()
    finger0 = NYUFingerRobot(config=config0)
    finger1 = NYUFingerRobot(config=config1)
    bullet_env.add_robot(finger0)
    bullet_env.add_robot(finger1)

    # add a plane
    plane = pybullet.loadURDF(os.path.join(
        current_path, "src/urdf/plane.urdf"))
    pybullet.resetBasePositionAndOrientation(
        plane, [0., 0., 0.], (0., 0., 0., 1.))
    pybullet.changeDynamics(
        plane, -1, lateralFriction=5., rollingFriction=0)

    # add a box
    box = pybullet.loadURDF(os.path.join(
        current_path, "src/urdf/box.urdf"))
    pybullet.resetBasePositionAndOrientation(
        box, [0., -0.2, 0.05], (0., 0., 0., 1.))
    pybullet.changeDynamics(
        box, -1, lateralFriction=0.5, spinningFriction=0.5)

    # set the robot initial state in theta
    finger0.reset_state(np.zeros(finger0.nq), np.zeros(finger0.nv))
    finger1.reset_state(np.zeros(finger1.nq), np.zeros(finger1.nv))

    for i in range(1000000000000000):

        # finger0 control -------------------------------------------------------
        q, dq = finger0.get_state()
        finger1.pin_robot.forwardKinematics(q)

        # if i % 1000 == 0:
        #     finger0.pin_robot.forwardKinematics(q)
        #     pose_trans = cal_forwardK(finger0.pin_robot, ID0).translation
        #     oj = cal_oriented_j(finger0.pin_robot, ID0, q)
        #     print(pose_trans)

        joint_torques0 = np.zeros(finger0.nq)
        finger0.send_joint_command(joint_torques0)
        # ------------------------------------------------------------------------

        # finger1 control -------------------------------------------------------
        q, dq = finger1.get_state()
        finger1.pin_robot.forwardKinematics(q)
        joint_torques1 = np.zeros(finger1.nv)
        finger1.send_joint_command(joint_torques1)
        # ------------------------------------------------------------------------

        # we send them to the robot and do one simulation step
        bullet_env.step(sleep=True)
