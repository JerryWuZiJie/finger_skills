'''
This run on real robot
'''
import os
import time
import sys

import numpy as np
import pinocchio as pin
import pybullet

from bullet_utils.env import BulletEnv
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from robot_properties_nyu_finger.wrapper import NYUFingerRobot

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


class ImpedanceControl():
    def __init__(self, P, D):
        # setup control parameters
        self.P = P
        self.D = D

    def cal_torque(self, x_ref, x_meas, dx_ref, dq, jacobian):
        # velocity part of jacobian
        Jov = jacobian[:3]

        # calculate torque
        dx_meas = Jov.dot(dq)
        des_force = self.P.dot(
            x_ref - x_meas) + self.D.dot(dx_ref - dx_meas)
        joint_torques = Jov.T.dot(des_force)

        return joint_torques


class EnvFingers:
    class Space:
        def __init__(self, shape):
            self.shape = (shape,)

    def __init__(self, des_pos=[0.2, 0, 0.05], render=True, ee0_id=ID0, ee1_id=ID1, dt=0.001):
        # desired position of the box
        self.des_pose = np.array(des_pos)
        # threshold for done condition
        self.threshold = 0.005  # meter, it is the square of distance

        # initialize impedance control
        self.control = ImpedanceControl(np.diag([50]*3), np.diag([1.]*3))

        # set initial box position
        self.box_default_pos = [0., 0., 0.05]
        self.box_default_ori = (0., 0., 0., 1.)
        # set initial joint angle
        self.finger_default_angle = np.zeros(3)  # [0., np.pi/4, -np.pi/4]
        # end effector id
        self.ee0_id = ee0_id
        self.ee1_id = ee1_id

        # whether to render the env and delay in step
        self._render = render
        # steps time interval
        self._dt = dt

        # init BulletEnv
        if render:
            self.bullet_env = BulletEnv(
                server=pybullet.GUI, dt=self._dt)  # BulletEnvWithGround()
            # set initial view point
            pybullet.resetDebugVisualizerCamera(.7, 0, -15, (0., 0., 0.2))
            # Disable the gui controller as we don't use them.
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.bullet_env = BulletEnv(
                server=pybullet.DIRECT, dt=self._dt)  # BulletEnvWithGround()

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

        if self._render:
            # add a ball as indication of desired position
            self.target = pybullet.loadURDF(
                os.path.join(current_path, "src/urdf/ball.urdf"))
            pybullet.resetBasePositionAndOrientation(
                self.target, self.des_pose, (0., 0., 0.5, 0.5))

        # space
        # (3 position + 3 velocity) * 2 fingers
        self.action_space = self.Space(12)
        # 3 position * 2 + 3 box position
        self.observation_space = self.Space(9)

    def reset(self):
        # set the robot initial state in theta
        self.finger0.reset_state(
            self.finger_default_angle, np.zeros(self.finger0.nv))
        self.finger1.reset_state(
            self.finger_default_angle, np.zeros(self.finger1.nv))

        # reset box position
        pybullet.resetBasePositionAndOrientation(
            self.boxid, self.box_default_pos, self.box_default_ori)

        # finger tip position
        q0, dq0 = self.finger0.get_state()
        self.finger0.pin_robot.forwardKinematics(q0)
        q1, dq1 = self.finger1.get_state()
        self.finger1.pin_robot.forwardKinematics(q1)
        ee0_pose = cal_forwardK(self.finger0.pin_robot,
                                self.ee0_id).translation
        ee1_pose = cal_forwardK(self.finger1.pin_robot,
                                self.ee1_id).translation

        # observation compose of finger tip position and box position
        observation = np.array([*ee0_pose, *ee1_pose, *self.des_pose])

        return observation

    def step(self, action):

        # update kinematic
        q0, dq0 = self.finger0.get_state()
        self.finger0.pin_robot.forwardKinematics(q0)
        q1, dq1 = self.finger1.get_state()
        self.finger1.pin_robot.forwardKinematics(q1)

        # finger tip position
        ee0_pose = cal_forwardK(self.finger0.pin_robot,
                                self.ee0_id).translation
        ee1_pose = cal_forwardK(self.finger1.pin_robot,
                                self.ee1_id).translation
        # finger tip oriented jacobian
        oj0 = cal_oriented_j(self.finger0.pin_robot, self.ee0_id, q0)
        oj1 = cal_oriented_j(self.finger1.pin_robot, self.ee1_id, q1)

        # calculate torque
        tau0 = self.control.cal_torque(
            action[:3], ee0_pose, action[3:6], dq0, oj0)
        tau1 = self.control.cal_torque(
            action[6:9], ee1_pose, action[9:], dq1, oj1)

        # send torque
        self.finger0.send_joint_command(tau0)
        self.finger1.send_joint_command(tau1)

        # delay the step if rendering
        self.bullet_env.step(sleep=self._render)

        # box position
        box_pose = pybullet.getBasePositionAndOrientation(self.boxid)[0]

        # observation compose of finger tip position and box position
        observation = np.array([*ee0_pose, *ee1_pose, *box_pose])

        # distance between box and desired location
        box_des = sum((box_pose-self.des_pose)**2)

        # reward is the negative of dist between box and des + fingers and box
        reward = -box_des * 10 + \
            (-sum((box_pose-ee0_pose)**2) -
             sum((box_pose-ee1_pose)**2))
        
        reward *= 10  # TODO: reward is not significant

        # done if box is close to desired location
        if box_des < self.threshold:
            done = True
            # TODO: uncomment on next train
            reward += 100  # make a big reward if solve the environment
        else:
            done = False

        # info
        info = {'reward: ': reward, }

        return observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pybullet.disconnect()


if __name__ == '__main__':

    env = EnvFingers(des_pos=[0.2, 0, 0.05], render=True)

    # position for finger 0
    des_pos = [0.0506947, 0.0594499, 0.05]  # init position
    des_pos_1 = np.array(des_pos)
    des_pos_1[0] += 0.1
    des_pos_1[2] += 0.1
    # position for finger 1
    des_pos_0 = np.array(des_pos_1)
    des_pos_0[0] = -des_pos_0[0]
    des_pos_0[1] = -des_pos_0[1]

    action = np.concatenate(
        (des_pos_0, np.zeros(3), des_pos_1, np.zeros(3)), axis=0)

    for i in range(3):
        done = False
        obs = env.reset()
        time.sleep(1)
        count = 0
        while not done:
            obs, rew, done, info = env.step(np.zeros(12))
            count += 1
            if count % 1000 == 0:
                print(info)
        print(i, 'DONE!!!')

    env.close()
