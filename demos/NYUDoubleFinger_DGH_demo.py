'''
This run on real robot
'''
import os
import numpy as np
import matplotlib.pylab as plt
import pinocchio as pin

from bullet_utils.env import BulletEnvWithGround
from bullet_utils.env import BulletEnv
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from robot_properties_nyu_finger.wrapper import NYUFingerRobot

from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController
# import dynamic_graph_manager_cpp_bindings


# function to calculate robot kinematics ----------------------
def cal_pose(pin_robot, id_ee):
    '''
    calculate position matrix (transition and rotation)
    '''
    # get frame id
    frame = pin_robot.model.getFrameId(id_ee)

    # get pose
    return pin.updateFramePlacement(pin_robot.model, pin_robot.data, frame)


def cal_oriented_j(pin_robot, id_ee, q):
    '''
    calculate oriented jacobian of the end effector
    '''
    # get frame id
    frame = pin_robot.model.getFrameId(id_ee)

    # get pose
    pose = pin.updateFramePlacement(pin_robot.model, pin_robot.data, frame)

    # get oriented jacobian
    body_jocobian = pin.computeFrameJacobian(
        pin_robot.model, pin_robot.data, q, frame)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = pose.rotation
    Ad[3:, 3:] = pose.rotation

    return Ad @ body_jocobian


def cal_inverseK():
    # TODO: given position, return desired angle
    pass


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
    def __init__(self, head, id_ee, param0, param1, control, total_time, dt=0.001):
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
        self.control = control(param0, param1)
        # hold still in the end of simulation
        self.hold_control = PDControl(4, .4)
        self.hold_time = 1  # second

        # setup steps
        self.total_time = total_time
        self.dt = dt
        self.current_step = 0

    def set_target(self, des_pos, des_vel=None):
        self.des_pos = des_pos
        if des_vel is None:
            self.des_vel = np.zeros(self.robot.nv)
        else:
            self.des_vel = des_vel

    def warmup(self, thread):
        pass

    def run(self, thread):
        # update robot kinematics
        self.robot.framesForwardKinematics(self.joint_positions)

        # switch to PD control and hold still for last seconds
        self.current_step += 1
        if self.current_step >= int((self.total_time-self.hold_time)/self.dt):
            if self.current_step == int((self.total_time-self.hold_time)/self.dt):
                # set current position (copy of joint pos) as dev_pos and 0 as des_vel
                self.set_target(np.array(self.joint_positions))
            self.tau = self.hold_control.cal_torque(
                self.des_pos, self.joint_positions, self.des_vel, self.joint_velocities)
            self.head.set_control('ctrl_joint_torques', self.tau)
            return False
        return True


class PDController(Controller):
    def __init__(self, head, id_ee, param0, param1, total_time, des_pos, des_vel=None):
        super().__init__(head, id_ee, param0, param1, PDControl, total_time)
        self.set_target(des_pos, des_vel)

    def run(self, thread):
        if super().run(thread):
            self.tau = self.control.cal_torque(
                self.des_pos, self.joint_positions, self.des_vel, self.joint_velocities)
            self.head.set_control('ctrl_joint_torques', self.tau)


class VelocityController(Controller):
    def __init__(self, head, id_ee, param0, param1, total_time, center, radius, speed=np.pi):
        super().__init__(head, id_ee, param0, param1, VelocityControl, total_time)

        # calculate circle locus
        self.locus = self.circular_locus(
            center, radius, total_time-self.hold_time, self.dt, speed)
        self.set_target(*self.locus[self.current_step])

    @staticmethod
    def circular_locus(center, radius, total_time, dt, w):
        '''
        calculate desire circular locus in xz plane
        '''
        length = int(total_time/dt) + \
            5  # switch controller has 1 extra run, +5 to ensure no index error
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
            self.set_target(*self.locus[self.current_step])


class ImpedanceController(Controller):
    def __init__(self, head, id_ee, param0, param1, total_time, des_pos, des_vel=None):
        super().__init__(head, id_ee, param0, param1, ImpedanceControl, total_time)
        self.set_target(des_pos, des_vel)

    def run(self, thread):
        if super().run(thread):
            # calculate position and oriented jacobian
            pose_trans = cal_pose(self.robot, self.id).translation
            oj = cal_oriented_j(self.robot, self.id, self.joint_positions)

            self.tau = self.control.cal_torque(
                self.des_pos, pose_trans, self.des_vel, self.joint_velocities, oj)
            self.head.set_control('ctrl_joint_torques', self.tau)


def choose_controller(finger, control, head, id, time):
    '''
    choose controller based on finger, return controller and q, dq
    '''
    if finger == 0:  # first finger
        if control == 0:
            # finger0 PD
            P = np.array([4,4,3])
            D = np.array([.5,.4,.2])
            ctrl = PDController(head, id, P, D, time, np.array([0,0,np.pi/2]))
            q = np.array(NYUFingerDoubleConfig0.initial_configuration)
            dq = np.array(NYUFingerDoubleConfig0.initial_velocity)
        elif control == 1:
            # finger0 velocity
            gain = 1.
            D = np.array([0.3, 0.3, 0.3])
            center = [-0.0506, -0.05945, 0.05]
            center[0] -= 0.1
            center[1] -= 0.05  # avoid two robot touching
            center[2] += 0.2
            radius = 0.04
            ctrl = VelocityController(head, id, gain, D, time, center, radius, np.pi*3)
            q = np.array([0., -0.5, 1])
            dq = np.array(NYUFingerDoubleConfig0.initial_velocity)
        else:
            # # finger0 impedance
            K = np.diag([50,50,10])
            D = np.diag([5,5,0])
            ctrl = ImpedanceController(head, id, K, D, time, np.array([-0.051-0.1, -0.059, 0.05+0.1]))
            q = np.array(NYUFingerDoubleConfig0.initial_configuration)
            dq = np.array(NYUFingerDoubleConfig0.initial_velocity)
    else:  # second finger
        if control == 0:
            # finger1 PD
            P = np.array([4,4,3])
            D = np.array([.5,.4,.2])
            ctrl = PDController(head, id, P, D, time, np.array([0,0,np.pi/2]))
            q = np.array(NYUFingerDoubleConfig1.initial_configuration)
            dq = np.array(NYUFingerDoubleConfig1.initial_velocity)
        elif control == 1:
            # finger1 velocity
            gain = 1.
            D = np.array([0.3, 0.3, 0.3])
            center = [0.0506947, 0.0594499, 0.05]
            center[0] += 0.1
            center[1] += 0.05  # avoid two robot touching
            center[2] += 0.2
            radius = 0.04
            ctrl = VelocityController(head, id, gain, D, time, center, radius, -np.pi*3)
            q = np.array([0., -0.5, 1])
            dq = np.array(NYUFingerDoubleConfig1.initial_velocity)
        else:
            # finger1 impedance
            K = np.diag([50,50,10])
            D = np.diag([5,5,0])
            ctrl = ImpedanceController(head, id, K, D, time, np.array([0.051+0.1, 0.059, 0.05+0.1]))
            q = np.array(NYUFingerDoubleConfig1.initial_configuration)
            dq = np.array(NYUFingerDoubleConfig1.initial_velocity)
    return ctrl, q, dq


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
    thread_head = ThreadHead(
        dt,  # dt.
        # Safety controllers.
        [HoldPDController(head0, 4, .4), HoldPDController(head1, 4, .4)],
        {'finger0': head0, 'finger1': head1},  # Heads to read / write from.
        [     # Utils.
            # ('vicon', SimVicon(['solo12/solo12']))  # not using it
        ],
        bullet_env  # Environment to step.
    )

    # setup controller
    ctrl0, q0, dq0 = choose_controller(0, 1, head0, id0, sim_time)
    thread_head.heads['finger0'].reset_state(q0, dq0)
    ctrl1, q1, dq1 = choose_controller(1, 2, head1, id1, sim_time)
    thread_head.heads['finger1'].reset_state(q1, dq1)

    # start simulation
    thread_head.switch_controllers((ctrl0, ctrl1))

    thread_head.start_streaming()
    thread_head.start_logging()

    thread_head.sim_run(int(sim_time/dt), sleep=True)

    thread_head.stop_streaming()
    thread_head.stop_logging()

    # Plot timing information.
    thread_head.plot_timing()


def main_real(sim_time, finger0_controller=0, finger1_controller=0):
    # setup some parameters that might not be used?
    sim_time = 5.0  # seconds
    dt = 0.001
    id0 = 'finger0_lower_to_tip_joint'
    id1 = 'finger1_lower_to_tip_joint'

    # Create the dgm communication and instantiate the controllers.
    path0 = os.path.join(NYUFingerDoubleConfig0().dgm_yaml_dir, 'dgm_parameters_nyu_finger_double_0.yaml')
    path1 = os.path.join(NYUFingerDoubleConfig1().dgm_yaml_dir, 'dgm_parameters_nyu_finger_double_1.yaml')
    head0 = dynamic_graph_manager_cpp_bindings.DGMHead(path0)
    head1 = dynamic_graph_manager_cpp_bindings.DGMHead(path1)
    head0.read()
    head1.read()
    print(head0.get_sensor('slider_positions'))
    print(head1.get_sensor('slider_positions'))

    # Create the safety controllers.
    hold_pd_controller0 = HoldPDController(head0, 3., 0.05, with_sliders=True)
    hold_pd_controller1 = HoldPDController(head1, 3., 0.05, with_sliders=False)

    # setup thread_head
    thread_head = ThreadHead(
        dt,
        [hold_pd_controller0, hold_pd_controller1],
        {'finger0': head0, 'finger1': head1},
        []
    )

    # setup controller
    ctrl0, q0, dq0 = choose_controller(0, 1, head0, id0, sim_time)
    thread_head.heads['finger0'].reset_state(q0, dq0)  # todo: no reset state, need to implement yourself
    ctrl1, q1, dq1 = choose_controller(1, 2, head1, id1, sim_time)
    thread_head.heads['finger1'].reset_state(q1, dq1)  # todo: no reset state, need to implement yourself

    # Start the parallel processing.
    thread_head.switch_controllers(ctrl0, ctrl1)
    thread_head.start()  # todo: it's an infinite loop, how should I stop?

    print("Finished controller setup")

    input("Wait for input to finish program.")
    # todo: main thread will wait for the subthread to finish, so how will this end the program?


if __name__ == '__main__':
    main_sim(5.0)
