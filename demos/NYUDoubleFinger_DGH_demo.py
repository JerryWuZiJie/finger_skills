'''
This run on real robot
'''
import os
import time
import sys
from threading import current_thread

import numpy as np
import matplotlib.pylab as plt
import pinocchio as pin

from bullet_utils.env import BulletEnvWithGround
from bullet_utils.env import BulletEnv
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from robot_properties_nyu_finger.wrapper import NYUFingerRobot

from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController
import dynamic_graph_manager_cpp_bindings


# param
PD_P = np.array([1.]*3)
PD_D = np.array([.1]*3)
VEL_P = 3
VEL_D = np.array([.3]*3)
IMP_P = np.diag([50, 50, 20])
IMP_D = np.diag([1., 1., 0])  # todo: it doesn't bounce at all
IMP_H_P = np.diag([50.]*3)
IMP_H_D = np.diag([.7]*3)

# function to calculate robot kinematics ----------------------


def cal_pose(pin_robot, id_ee):
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
    given desire position, return desire q'''
    # todo: calculate inverse kinematic

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


# controls for calculation ----------------------
class Control:
    def __init__(self, param0, param1):
        # setup control parameters, eg. P and D/gain and D/spring and damping const
        self.set_control(param0, param1)

    def set_control(self, param0, param1):
        # setup control parameters
        self.param0 = param0
        self.param1 = param1

    def cal_torque(self):
        # given desire position and velocity, calculate the torques for all joints
        raise NotImplementedError


class PDControl(Control):
    def cal_torque(self, q_ref, q, dq_ref, dq):
        # calculate error
        # the position error for all the joints (it's a 3D vector)
        error = q_ref - q
        d_error = dq_ref - dq  # the velocity error for all the joints

        # calculate torque
        joint_torques = self.param0 * error + self.param1 * d_error

        return joint_torques


class VelocityControl(Control):
    def cal_torque(self, x_ref, x_meas, dx_ref, q, dq, jacobian):
        # velocity part of jacobian
        Jov = jacobian[:3]
        # calculate inverse Jacobian
        if np.abs(q[2] < 0.05):
            Jov_inv = np.linalg.pinv(Jov + 1e-4*np.eye(3))
        else:
            Jov_inv = np.linalg.pinv(Jov)

        # calculate torque
        dx_des = dx_ref + self.param0 * (x_ref - x_meas)
        dq_des = Jov_inv @ dx_des
        d_error = dq_des - dq
        joint_torques = self.param1 * d_error

        return joint_torques


class ImpedanceControl(Control):
    def cal_torque(self, x_ref, x_meas, dx_ref, dq, jacobian):
        # velocity part of jacobian
        Jov = jacobian[:3]

        # calculate torque
        dx_meas = Jov.dot(dq)
        des_force = self.param0.dot(
            x_ref - x_meas) + self.param1.dot(dx_ref - dx_meas)
        joint_torques = Jov.T.dot(des_force)

        return joint_torques


# controllers for running ----------------------
class Controller:
    def __init__(self, head, id_ee, param0, param1, control, dt=0.001, hold_time=2):
        # setup head
        self.head = head
        # pos and vel is reference, will update automatically
        self.joint_positions = self.head.get_sensor('joint_positions')
        self.joint_velocities = self.head.get_sensor('joint_velocities')

        # setup pinocchio robot model
        self.id = id_ee
        if self.id == 'finger0_lower_to_tip_joint':
            self.robot = NYUFingerDoubleConfig0.buildRobotWrapper()
        else:
            self.robot = NYUFingerDoubleConfig1.buildRobotWrapper()

        # setup control
        if type(param0) != np.ndarray:
            param0 = np.array([param0]*self.robot.nq)
        if type(param1) != np.ndarray:
            param1 = np.array([param1]*self.robot.nv)
        self.control = control(param0, param1)
        # move robot to it's initial position
        self.position_control = ImpedanceControl(IMP_H_P, IMP_H_D)
        self.position_steps = int(hold_time/dt)

        # setup steps
        self.dt = dt
        self.current_step = 0

    def set_target(self, des_pos, des_vel=None):
        if type(des_pos) != np.ndarray:
            des_pos = np.array([des_pos] * 3)
        self.des_pos = des_pos
        if des_vel is None:
            self.des_vel = np.zeros(self.robot.nv)
        else:
            if type(des_vel) != np.ndarray:
                des_vel = np.array([des_vel] * 3)
            self.des_vel = des_vel

    def warmup(self, thread):
        pass

    def run(self, thread):
        # update robot kinematics
        self.robot.framesForwardKinematics(self.joint_positions)

        # use impedance control to make robot move to initial state
        if self.current_step <= self.position_steps:
            if type(self) != PDController:
                pose_trans = cal_pose(self.robot, self.id).translation
                oj = cal_oriented_j(self.robot, self.id, self.joint_positions)
                self.tau = self.position_control.cal_torque(
                    self.des_pos, pose_trans, np.zeros(self.robot.nq), self.joint_velocities, oj)
                self.head.set_control('ctrl_joint_torques', self.tau)

                self.current_step += 1
                return False
        self.current_step += 1
        return True


class PDController(Controller):
    def __init__(self, head, id_ee, param0, param1, des_pos, des_vel=None):
        super().__init__(head, id_ee, param0, param1, PDControl)
        self.set_target(des_pos, des_vel)

        # # calculate initial position in angle
        # self.init_pos = self.des_pos
        # self.init_vel = self.des_vel

    def run(self, thread):
        if super().run(thread):
            self.tau = self.control.cal_torque(
                self.des_pos, self.joint_positions, self.des_vel, self.joint_velocities)
            self.head.set_control('ctrl_joint_torques', self.tau)


class VelocityController(Controller):
    def __init__(self, head, id_ee, param0, param1, center, radius, speed=np.pi):
        super().__init__(head, id_ee, param0, param1, VelocityControl)

        # calculate circle locus
        self.locus = self.circular_locus(
            center, radius, speed, self.dt)
        self.set_target(*self.locus[0])

        # # calculate initial position in angle
        # self.init_pos = cal_inverseK(
        #     self.robot, self.id, self.des_pos, self.joint_positions)
        # self.init_vel = np.zeros(self.robot.nv)

    @staticmethod
    def circular_locus(center, radius, w, dt):
        '''
        calculate desire circular locus in xz plane
        '''
        # 2*np.pi, w*dt

        length = abs(int(2*np.pi / (w * dt)))
        locus = []
        for i in range(length):
            t = dt * i
            x_ref = np.array([center[0] + radius * np.sin(w * t),
                              center[1],
                              center[2] + radius * np.cos(w * t)])
            dx_ref = np.array([radius * w * np.cos(w * t),
                               0.,
                               -radius * w * np.sin(w * t)])
            locus.append((x_ref, dx_ref))

        return locus

    def run(self, thread):
        if super().run(thread):
            # calculate position and oriented jacobian
            pose_trans = cal_pose(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            self.tau = self.control.cal_torque(
                self.des_pos, pose_trans, self.des_vel, self.joint_positions, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)

            # update target
            index = (self.current_step-self.position_steps) % len(self.locus)
            self.set_target(*self.locus[index])


class ImpedanceController(Controller):
    def __init__(self, head, id_ee, param0, param1, des_pos, des_vel=None):
        super().__init__(head, id_ee, param0, param1, ImpedanceControl)
        self.set_target(des_pos, des_vel)

        # # calculate initial position in angle
        # self.init_pos = cal_inverseK(
        #     self.robot, self.id, self.des_pos, self.joint_positions)
        # self.init_vel = np.zeros(self.robot.nv)

    def run(self, thread):
        if super().run(thread):
            # calculate position and oriented jacobian
            pose_trans = cal_pose(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            self.tau = self.control.cal_torque(
                self.des_pos, pose_trans, self.des_vel, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)


def choose_controller(finger, control, head, id):
    '''
    choose controller based on finger, return controller and q, dq
    '''
    center = [0.0506947, 0.0594499, 0.05]  # init position
    center[0] += 0.1
    center[1] += 0.05  # avoid two robot touching
    center[2] += 0.1
    radius = 0.02

    if finger == 0:  # first finger
        if control == 0:
            # finger0 PD
            ctrl = PDController(head, id, PD_P, PD_D,
                                np.array([0, 0, np.pi/3]))
        elif control == 1:
            # finger0 velocity
            center0 = center
            center0[0] = -center[0]
            center0[1] = -center[1]
            ctrl = VelocityController(
                head, id, VEL_P, VEL_D, center0, radius, np.pi*3)
        else:
            # # finger0 impedance
            ctrl = ImpedanceController(
                head, id, IMP_P, IMP_D, np.array([-0.0506-0.1, -0.05945, 0.05+0.1]))
    else:  # second finger
        if control == 0:
            # finger1 PD
            ctrl = PDController(head, id, PD_P, PD_D,
                                np.array([0, 0, np.pi/3]))
        elif control == 1:
            # finger1 velocity
            center1 = center
            ctrl = VelocityController(
                head, id, VEL_P, VEL_D, center1, radius, -np.pi*3)
        else:
            # finger1 impedance
            ctrl = ImpedanceController(
                head, id, IMP_P, IMP_D, np.array([0.051+0.1, 0.059, 0.05+0.1]))
    return ctrl


if __name__ == '__main__':
    # setup some parameters
    SIMULATION = 0
    FINGER0_CONTROLLER = 2
    FINGER1_CONTROLLER = 2
    FINGER0_ONLY = False
    ID0 = 'finger0_lower_to_tip_joint'
    ID1 = 'finger1_lower_to_tip_joint'
    DT = 0.001

    # setup head
    if SIMULATION:
        # init BulletEnv and setup robot ----------------------
        bullet_env = BulletEnv()  # BulletEnvWithGround()

        # Create a robot instance. This initializes the simulator as well.
        config0 = NYUFingerDoubleConfig0()
        config1 = NYUFingerDoubleConfig1()
        finger0 = NYUFingerRobot(config=config0)
        finger1 = NYUFingerRobot(config=config1)
        id0 = 'finger0_lower_to_tip_joint'
        id1 = 'finger1_lower_to_tip_joint'
        bullet_env.add_robot(finger0)
        bullet_env.add_robot(finger1)
        head0 = SimHead(finger0, vicon_name='solo12', with_sliders=False)
        head1 = SimHead(finger1, vicon_name='solo12', with_sliders=False)
        head_dict = {'default': head0} if FINGER0_ONLY else {
            'finger0': head0, 'finger1': head1}
    else:
        # Create the dgm communication and instantiate the controllers.
        path0 = os.path.join(NYUFingerDoubleConfig0().dgm_yaml_dir,
                             'dgm_parameters_nyu_finger_double_0.yaml')
        path1 = os.path.join(NYUFingerDoubleConfig1().dgm_yaml_dir,
                             'dgm_parameters_nyu_finger_double_1.yaml')
        head0 = dynamic_graph_manager_cpp_bindings.DGMHead(path0)
        head1 = dynamic_graph_manager_cpp_bindings.DGMHead(path1)
        head0.read()
        head1.read()
        head_dict = {'default': head0} if FINGER0_ONLY else {
            'finger0': head0, 'finger1': head1}
        bullet_env = None

    # setup safety controllers to hold still when error occur
    safety_pd_controller0 = HoldPDController(
        head0, 3., 0.05, with_sliders=False)
    safety_pd_controller1 = HoldPDController(
        head1, 3., 0.05, with_sliders=False)
    safety_controllers = safety_pd_controller0 if FINGER0_ONLY else [
        safety_pd_controller0, safety_pd_controller1]

    # setup run controllers for running
    ctrl0 = choose_controller(0, FINGER0_CONTROLLER, head0, ID0)
    ctrl1 = choose_controller(1, FINGER1_CONTROLLER, head1, ID1)
    controllers = ctrl0 if FINGER0_ONLY else [ctrl0, ctrl1]

    # setup hold controllers for the end
    des_angle = np.zeros(3)
    end_ctrl0 = PDController(head0, ID0, PD_P, PD_D, des_angle)
    end_ctrl1 = PDController(head1, ID1, PD_P, PD_D, des_angle)
    end_controllers = [end_ctrl0, end_ctrl1]

    # setup thread_head
    thread_head = ThreadHead(
        DT,
        safety_controllers,
        head_dict,
        [],  # utils, no for now
        bullet_env
    )

    # call this function in ipython if try to stop the run
    def stop(wait_time=1):
        print('\n', '-'*50, sep='')
        print('Apply PD control to turn robot to origin pos')
        print('-'*50)

        # Switch to end_controller
        thread_head.switch_controllers(end_controllers)

        # run for a bit and then stop
        time.sleep(wait_time)
        thread_head.run_loop = False

        # set control torque to 0
        for head in head_dict.values():
            head.set_control('ctrl_joint_torques', np.array([0.]*3))
            head.write()
        print('-'*10, 'exit', '-'*10)
        if SIMULATION:
            thread_head.plot_timing()
        # sys.exit(0)  # this raise error in ipython

    # Start the parallel processing.
    thread_head.switch_controllers(controllers)
    thread_head.start()

    # logging syntax in ipython: thread_head.start_logging(); time.sleep(1); thread_head.stop_logging()
