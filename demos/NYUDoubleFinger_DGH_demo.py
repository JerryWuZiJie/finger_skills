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
PD_0 =np.array([3.]*3)
PD_1 = np.array([.1]*3)
VEL_0 = 1
VEL_1 = np.array([.3]*3)
IMP_0 = np.diag([50, 50, 10])
IMP_1 = np.diag([5, 5, 0])
IMP_H_0 = np.diag([10]*3)
IMP_H_1 = np.diag([5]*3)

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
    def __init__(self, head, id_ee, param0, param1, control, dt=0.001, hold_time=5):
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
        self.position_control = ImpedanceControl(IMP_H_0, IMP_H_1)
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
    if finger == 0:  # first finger
        if control == 0:
            # finger0 PD
            ctrl = PDController(head, id, PD_0, PD_1, np.array([0, 0, np.pi/3]))
        elif control == 1:
            # finger0 velocity
            center = [-0.0506, -0.05945, 0.05]
            center[0] -= 0.1
            center[1] -= 0.05  # avoid two robot touching
            center[2] += 0.15
            radius = 0.04
            ctrl = VelocityController(
                head, id, VEL_0, VEL_1, center, radius, np.pi*3)
        else:
            # # finger0 impedance
            ctrl = ImpedanceController(
                head, id, IMP_0, IMP_1, np.array([-0.0506-0.1, -0.05945, 0.05+0.1]))
    else:  # second finger
        if control == 0:
            # finger1 PD
            ctrl = PDController(head, id, PD_0, PD_1, np.array([0, 0, np.pi/3]))
        elif control == 1:
            # finger1 velocity
            center = [0.0506947, 0.0594499, 0.05]
            center[0] += 0.1
            center[1] += 0.05  # avoid two robot touching
            center[2] += 0.15
            radius = 0.04
            ctrl = VelocityController(
                head, id, VEL_0, VEL_1, center, radius, -np.pi*3)
        else:
            # finger1 impedance
            ctrl = ImpedanceController(
                head, id, IMP_0, IMP_1, np.array([0.051+0.1, 0.059, 0.05+0.1]))
    return ctrl


def main_sim(sim_time, finger0_controller=0, finger1_controller=0):
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

    # setup simulation and thread_head ----------------------
    dt = 0.001

    head0 = SimHead(finger0, vicon_name='solo12', with_sliders=False)
    head1 = SimHead(finger1, vicon_name='solo12', with_sliders=False)
    head_dict = {'finger0': head0, 'finger1': head1}

    thread_head = ThreadHead(
        dt,  # dt.
        # Safety controllers.
        [HoldPDController(head0, 4, .4), HoldPDController(head1, 4, .4)],
        head_dict,  # Heads to read / write from.
        [     # Utils.
            # ('vicon', SimVicon(['solo12/solo12']))  # not using it
        ],
        bullet_env  # Environment to step.
    )

    # setup controller
    ctrl0 = choose_controller(0, finger0_controller, head0, id0)
    ctrl1 = choose_controller(1, finger1_controller, head1, id1)

    try:
        # start simulation
        thread_head.switch_controllers((ctrl0, ctrl1))

        # thread_head.start_streaming()
        # thread_head.start_logging()

        # thread_head.sim_run(int(sim_time/dt), sleep=True)
        thread_head.start()
        thread_head.join()

        # thread_head.stop_streaming()
        # thread_head.stop_logging()

        # Plot timing information.
        thread_head.plot_timing()
    except KeyboardInterrupt:
        print('\n', '-'*50, sep='')
        print('ctrl c pressed, apply PD control to turn robot to origin pos')
        print('-'*50)
        thread_head.run_loop = False

    # set up end PD control
    end_ctrl = PDControl(PD_0, PD_1)
    for _ in range(1000):
        for head in head_dict.values():
            head.read()
            q = head.get_sensor('joint_positions')
            dq = head.get_sensor('joint_velocities')
            tau = end_ctrl.cal_torque(
                np.array([0.]*3), q, np.array([0.]*3), dq)
            head.set_control('ctrl_joint_torques', tau)
            head.write()

        # Step the actual simulation.
        bullet_env.step(sleep=True)

    print('-'*10, 'exit', '-'*10)


def main_real(num=2, finger0_controller=0, finger1_controller=0):
    # setup some parameters that might not be used?
    dt = 0.001
    id0 = 'finger0_lower_to_tip_joint'
    id1 = 'finger1_lower_to_tip_joint'

    # Create the dgm communication and instantiate the controllers.
    path0 = os.path.join(NYUFingerDoubleConfig0().dgm_yaml_dir,
                         'dgm_parameters_nyu_finger_double_0.yaml')
    path1 = os.path.join(NYUFingerDoubleConfig1().dgm_yaml_dir,
                         'dgm_parameters_nyu_finger_double_1.yaml')
    head0 = dynamic_graph_manager_cpp_bindings.DGMHead(path0)
    head1 = dynamic_graph_manager_cpp_bindings.DGMHead(path1)
    head0.read()
    head1.read()

    # Create the safety controllers.
    hold_pd_controller0 = HoldPDController(head0, 3., 0.05, with_sliders=False)
    hold_pd_controller1 = HoldPDController(head1, 3., 0.05, with_sliders=False)
    holders = hold_pd_controller0 if num!=2 else [hold_pd_controller0, hold_pd_controller1]
    head_dict = head0 if num!=2 else {'finger0': head0, 'finger1': head1}


    # setup controller
    ctrl0 = choose_controller(0, finger0_controller, head0, id0)
    ctrl1 = choose_controller(1, finger1_controller, head1, id1)
    controllers = ctrl0 if num!=2 else [ctrl0, ctrl1]

    # setup thread_head
    thread_head = ThreadHead(
        dt,
        holders,
        head_dict,
        []
    )


    # Start the parallel processing.
    thread_head.switch_controllers(controllers)
    try:
        thread_head.start()
        thread_head.join()
    except KeyboardInterrupt:
        print('\n', '-'*50, sep='')
        print('ctrl c pressed, apply PD control to turn robot to origin pos')
        print('-'*50)
        thread_head.run_loop = False

    # set up end PD control
    end_ctrl = PDControl(PD_0, PD_1)
    for _ in range(1000):
        for head in head_dict.values():
            head.read()
            q = head.get_sensor('joint_positions')
            dq = head.get_sensor('joint_velocities')
            tau = end_ctrl.cal_torque(
                np.array([0.]*3), q, np.array([0.]*3), dq)
            head.set_control('ctrl_joint_torques', tau)
            head.write()
            time.sleep(dt)

    print('-'*10, 'exit', '-'*10)


if __name__ == '__main__':

    main_sim(8, 1, 1)
    # main_real(1, 1, 1)
