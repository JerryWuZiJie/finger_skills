'''
This run on real robot
'''
import os
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

try:
    import dynamic_graph_manager_cpp_bindings
except ModuleNotFoundError as exc:
    print(exc)

# setup some constants
SIMULATION = Trueg  # simulation or run on real robot
SHOW_TIMING = False  # show timing log at the end
# control for finger0 and finger1.    0: PD; 1: velocity; 2: impedance
FINGER0_CONTROLLER = 0
FINGER1_CONTROLLER = 0
FINGER0_ONLY = False  # if true, only control finger0
ID0 = 'finger0_lower_to_tip_joint'  # finger0 end effector id
ID1 = 'finger1_lower_to_tip_joint'  # finger1 end effector id
DT = 0.001  # step time
# setup control parameters
PD_P = np.array([1.]*3)
PD_D = np.array([.1]*3)
VEL_P = 5
VEL_D = np.array([.1]*3)
IMP_P = np.diag([50, 50, 50])
IMP_D = np.diag([1., 0., 1.])
IMP_H_P = np.diag([100.]*3)  # TODO: test on real robot carefully
IMP_H_D = np.diag([1]*3)  # TODO: test on real robot carefully

# final position
# for PD control
PD_DES = np.array([0, np.pi/3, -np.pi/3])

center = [0.0506947, 0.0594499, 0.05]  # init position
# for velocity control
center_vel = np.array(center)
center_vel[0] += 0.1
center_vel[1] += 0.05  # avoid two robot touching
center_vel[2] += 0.15
speed = np.pi*3
radius = 0.08
# for impedance control
center_imp = np.array(center)
center_imp[0] += 0.1
center_imp[2] += 0.1


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


def interpolation_trajectory(th_init, th_goal, movement_duration, dt, des_vel):
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


def circular_trajectory(center, radius, w, dt):
    '''
    calculate desire circular trajectory in xz plane
    '''
    # 2*np.pi, w*dt

    length = abs(int(2*np.pi / (w * dt)))
    trajectory = []
    for i in range(length):
        t = dt * i
        x_ref = np.array([center[0] + radius * np.sin(w * t),
                          center[1],
                          center[2] + radius * np.cos(w * t)])
        dx_ref = np.array([radius * w * np.cos(w * t),
                           0.,
                           -radius * w * np.sin(w * t)])
        trajectory.append((x_ref, dx_ref))

    return trajectory


# controls for calculation ----------------------
class Control:
    def __init__(self, P, D):
        # setup control parameters, eg. P and D/gain and D/spring and damping const
        self.set_control(P, D)

    def set_control(self, P, D):
        # setup control parameters
        self.P = P
        self.D = D

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
        joint_torques = self.P * error + self.D * d_error

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
        dx_des = dx_ref + self.P * (x_ref - x_meas)
        dq_des = Jov_inv @ dx_des
        d_error = dq_des - dq
        joint_torques = self.D * d_error

        return joint_torques


class ImpedanceControl(Control):
    def cal_torque(self, x_ref, x_meas, dx_ref, dq, jacobian):
        # velocity part of jacobian
        Jov = jacobian[:3]

        # calculate torque
        dx_meas = Jov.dot(dq)
        des_force = self.P.dot(
            x_ref - x_meas) + self.D.dot(dx_ref - dx_meas)
        joint_torques = Jov.T.dot(des_force)

        return joint_torques


# controllers for running ----------------------
class Controller:
    def __init__(self, head, id_ee, P, D, control, dt=0.001, trans_coeff=10):
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
        if type(P) != np.ndarray:
            P = np.array([P]*self.robot.nq)
        if type(D) != np.ndarray:
            D = np.array([D]*self.robot.nv)
        self.control = control(P, D)

        # setup step and robot
        self.dt = dt
        self.current_step = 0
        self.robot.framesForwardKinematics(self.joint_positions)

        # setup init_control
        self.init_control = ImpedanceControl(IMP_H_P, IMP_H_D)
        self.trans_coeff = trans_coeff

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

    def reset_trajectory(self):
        pose_ee = cal_forwardK(self.robot, self.id).translation

        # transit time depends on diff on largest angle
        self.transit_time = max(abs(self.des_pos - pose_ee)) * self.trans_coeff

        # calculate interpolation trajectory
        if self.transit_time and int(self.transit_time/self.dt):
            self.init_trajectory = interpolation_trajectory(
                pose_ee, self.des_pos, self.transit_time, self.dt, self.des_vel)
        else:  # the robot is already in position
            self.init_trajectory = [(self.des_pos, self.des_vel)]

    def warmup(self, thread):
        self.robot.framesForwardKinematics(self.joint_positions)
        self.reset_trajectory()
        self.current_step = 0

    def run(self, thread):
        # update robot kinematics
        self.robot.framesForwardKinematics(self.joint_positions)
        self.current_step += 1

        # init control mode
        if self.current_step < len(self.init_trajectory):
            temp_pos, temp_vel = self.init_trajectory[self.current_step]

            # calculate position and oriented jacobian
            pose_ee = cal_forwardK(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            # send torque
            self.tau = self.init_control.cal_torque(
                temp_pos, pose_ee, temp_vel, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)

            return False
        else:
            return True


class PDController(Controller):
    '''
    control completely by PD, no impedance control
    '''

    def __init__(self, head, id_ee, PD_P, PD_D, des_pos_p, des_vel_p=None):
        super().__init__(head, id_ee, PD_P, PD_D, PDControl)

        des_pos = des_pos_p
        des_vel = des_vel_p
        self.set_target(des_pos, des_vel)

    def reset_trajectory(self):
        # transit time depends on diff on largest angle
        self.transit_time = max(abs(self.des_pos - self.joint_positions))
        # calculate interpolation trajectory
        if self.transit_time and int(self.transit_time/self.dt):
            self.init_trajectory = interpolation_trajectory(
                self.joint_positions, self.des_pos,
                self.transit_time, self.dt, self.des_vel)
        else:
            self.init_trajectory = [(self.des_pos, self.des_vel)]

    def run(self, thread):
        # update robot kinematics
        self.robot.framesForwardKinematics(self.joint_positions)
        self.current_step += 1

        # init control mode
        if self.current_step < len(self.init_trajectory):
            temp_pos, temp_vel = self.init_trajectory[self.current_step]
        else:
            temp_pos, temp_vel = self.init_trajectory[-1]

        # calculate position and oriented jacobian
        pose_ee = cal_forwardK(self.robot, self.id).translation
        oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

        # send torque
        self.tau = self.control.cal_torque(
            temp_pos, self.joint_positions,
            temp_vel, self.joint_velocities)
        self.head.set_control('ctrl_joint_torques', self.tau)


class NotUsePDController(Controller):
    def __init__(self, head, id_ee, PD_P, PD_D, des_pos_p, des_vel_p=None):
        super().__init__(head, id_ee, PD_P, PD_D, PDControl)

        # set the target
        self.set_target(des_pos_p, des_vel_p)

    def set_target(self, des_pos_p, des_vel_p):
        # store des p and v in polar coord
        self.des_pos_p = des_pos_p
        self.des_vel_p = des_vel_p if des_vel_p else np.zeros(self.robot.nv)

        # store des p and v in cartesian coord
        self.robot.framesForwardKinematics(self.des_pos_p)
        des_pos = cal_forwardK(self.robot, self.id).translation
        super().set_target(des_pos)  # TODO: assume vel is 0
        # set robot back to its current state
        self.robot.framesForwardKinematics(self.joint_positions)

    def run(self, thread):
        if super().run(thread):
            temp_pos, temp_vel = self.init_trajectory[-1]
            # send torque
            self.tau = self.control.cal_torque(
                temp_pos, self.joint_positions,
                temp_vel, self.joint_velocities)
            self.head.set_control('ctrl_joint_torques', self.tau)


class VelocityController(Controller):
    def __init__(self, head, id_ee, param0, param1, center, radius, speed=np.pi, dt=0.001, trans_coeff=10):
        super().__init__(head, id_ee, param0, param1, VelocityControl, dt, trans_coeff)

        # calculate circle path
        self.path = circular_trajectory(
            center, radius, speed, self.dt)
        self.set_target(*self.path[0])

    def run(self, thread):
        if super().run(thread):
            # update pos and vel
            index = (self.current_step -
                     len(self.init_trajectory)) % len(self.path)
            temp_pos, temp_vel = self.path[index]

            # calculate position and oriented jacobian
            pose_ee = cal_forwardK(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            # send torque
            self.tau = self.control.cal_torque(
                temp_pos, pose_ee, temp_vel, self.joint_positions, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)


class ImpedanceController(Controller):
    def __init__(self, head, id_ee, param0, param1, des_pos, des_vel=None, dt=0.001, trans_coeff=10):
        super().__init__(head, id_ee, param0, param1, ImpedanceControl, dt, trans_coeff)

        self.set_target(des_pos, des_vel)

    def run(self, thread):
        if super().run(thread):
            temp_pos, temp_vel = self.init_trajectory[-1]

            # calculate position and oriented jacobian
            pose_ee = cal_forwardK(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            # send torque
            self.tau = self.control.cal_torque(
                temp_pos, pose_ee, temp_vel, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)


def choose_controller(finger, control, head, id):
    '''
    choose controller based on finger, return controller and q, dq
    '''

    if finger == 0:  # first finger
        if control == 0:
            # finger0 PD
            ctrl = PDController(head, id, PD_P, PD_D, PD_DES)
        elif control == 1:
            # finger0 velocity
            center0 = np.array(center_vel)
            center0[0] = -center_vel[0]
            center0[1] = -center_vel[1]
            ctrl = VelocityController(
                head, id, VEL_P, VEL_D, center0, radius, speed)
        else:
            # finger0 impedance
            center_imp0 = np.array(center_imp)
            center_imp0[0] = -center_imp0[0]
            center_imp0[1] = -center_imp0[1]
            ctrl = ImpedanceController(
                head, id, IMP_P, IMP_D, center_imp0)
    else:  # second finger
        if control == 0:
            # finger1 PD
            ctrl = PDController(head, id, PD_P, PD_D, PD_DES)
        elif control == 1:
            # finger1 velocity
            center1 = np.array(center_vel)
            ctrl = VelocityController(
                head, id, VEL_P, VEL_D, center1, radius, -speed)
        else:
            # finger1 impedance
            center_imp1 = np.array(center_imp)
            ctrl = ImpedanceController(
                head, id, IMP_P, IMP_D, center_imp1)
    return ctrl


if __name__ == '__main__':

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

        # add a plane
        plane = pybullet.loadURDF("urdf/plane.urdf")
        pybullet.resetBasePositionAndOrientation(
            plane, [0., 0., 0.], (0., 0., 0., 1.))
        pybullet.changeDynamics(
            plane, -1, lateralFriction=5., rollingFriction=0)
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

    # setup trajectory to grab a box
    move_in = 0.04  # formula: mg=uF, Kx=F => x = mg/u/K = 0.5/2*9.8/0.5/100 approx 0.049 + 0.003 margin
    lift_up = 0.1
    shift = 0
    set_point_0 = [
        np.array([-0.15, 0.0, 0.1]),
        np.array([-0.075+move_in, 0.0, 0.08]),
        np.array([-0.075+move_in, 0.0, 0.08+lift_up]),
        np.array([-0.075+move_in, 0.0, 0.08]),
        np.array([-0.15, 0.0, 0.1]),
    ]
    set_point_1 = [
        np.array([0.15, 0.0, 0.1]),
        np.array([0.075-move_in, 0.0, 0.08]),
        np.array([0.075-move_in, 0.0, 0.08+lift_up]),
        np.array([0.075-move_in, 0.0, 0.08]),
        np.array([0.15, 0.0, 0.1]),
    ]
    temp_speed = 15
    grab_ctrl0 = ImpedanceController(
        head0, ID0, IMP_H_P, IMP_H_D, set_point_0[0], trans_coeff=temp_speed)
    grab_ctrl1 = ImpedanceController(
        head1, ID1, IMP_H_P, IMP_H_D, set_point_1[0], trans_coeff=temp_speed)
    grab_controllers = [grab_ctrl0, grab_ctrl1]

    # setup thread_head
    thread_head = ThreadHead(
        DT,
        safety_controllers,
        head_dict,
        [],  # utils, no for now
        bullet_env
    )

    def grab_object():

        # if SIMULATION:
        #     # add box to simulation
        #     box = pybullet.loadURDF("urdf/box.urdf")
        #     pybullet.resetBasePositionAndOrientation(
        #         box, [0., 0., 0.], (0., 0., 0., 1.))
        #     pybullet.changeDynamics(
        #         box, -1, lateralFriction=0.5, spinningFriction=0.5)

        for i in range(len(set_point_0)):
            grab_ctrl0.set_target(set_point_0[i])
            grab_ctrl1.set_target(set_point_1[i])
            thread_head.switch_controllers(grab_controllers)
            time.sleep(max(grab_ctrl0.transit_time,
                       grab_ctrl1.transit_time) + 0.5)

    # call this function in ipython if try to stop the run
    def stop(wait_time=1):
        print('\n', '-'*50, sep='')
        print('Apply PD control to turn robot to origin pos')
        print('-'*50)

        # Switch to end_controller
        thread_head.switch_controllers(end_controllers)

        # set the trajectory for the PD controller
        for control in end_controllers:
            wait_time = max(wait_time, control.transit_time)
        wait_time += 0.5

        # run for wait_time and then stop
        time.sleep(wait_time)
        thread_head.run_loop = False

        # set control torque to 0
        for head in head_dict.values():
            head.set_control('ctrl_joint_torques', np.array([0.]*3))
            head.write()
        print('-'*22, 'exit', '-'*22)
        if SHOW_TIMING:
            thread_head.plot_timing()
        # sys.exit(0)  # this raise error in ipython

    def s():
        # emergency stop
        thread_head.run_loop = False

        # set control torque to 0
        for head in head_dict.values():
            head.set_control('ctrl_joint_torques', np.array([0.]*3))
            head.write()

    def grab_in_thread():
        grab_object()

    grab_thread = threading.Thread(target=grab_in_thread)

    # Start the parallel processing.
    thread_head.switch_controllers(controllers)
    thread_head.start()

    for control in controllers:
        wait_time = max(0, control.transit_time)
    time.sleep(wait_time)

    grab_thread.start()

    # logging syntax in ipython: thread_head.start_logging(); time.sleep(1); thread_head.stop_logging()
