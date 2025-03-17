import rospy
import numpy as np
from scipy.interpolate import make_interp_spline
from geometry_msgs.msg import PoseStamped, Twist

class Guidance: 
    def __init__(self, x0=0, y0=0): 
        # Giving way points in the pool I can generate a trajectory and then
        # calculate the points for each time stamp 

        self.x0 = x0 
        self.y0 = y0

        #Testin with a straight line
        #self.wpt_pos_x = [0.0, 0.3, 0.6, 0.9, 1, 1.2]
        #self.wpt_pos_y = [0, 0, 0, 0, 0, 0]

        #self.wpt_pos_x = [self.x0, self.x0+1.0, self.x0+2.0, self.x0+2.0, self.x0+2.0, self.x0+2.0]
        self.wpt_pos_x = [self.x0, self.x0, self.x0, self.x0, self.x0, self.x0]
        #self.wpt_pos_y = [self.y0, self.y0, self.y0, self.y0+0.5, self.y0+1.0, self.y0+1.0]
        self.wpt_pos_y = [self.y0, self.y0, self.y0, self.y0, self.y0, self.y0]

        self.wpt_time = [0, 10, 20, 30, 40, 50] ## go there in 50 seconds 
        self.h = 0.1  # Time step  (10Hz)
        self.time = np.arange(0, max(self.wpt_time), self.h)

        # Generate reference trajectories using B-splines
        self.x_d_spline = make_interp_spline(self.wpt_time, self.wpt_pos_x, k=3)
        self.y_d_spline = make_interp_spline(self.wpt_time, self.wpt_pos_y, k=3)

        self.x_d = self.x_d_spline(self.time)
        self.y_d = self.y_d_spline(self.time)
        
        # Compute first derivatives (velocities)
        self.dx_d = self.x_d_spline.derivative()(self.time)
        self.dy_d = self.y_d_spline.derivative()(self.time)

        # Compute second derivatives (for yaw rate calculation)
        self.ddx_d = self.x_d_spline.derivative(nu=2)(self.time)
        self.ddy_d = self.y_d_spline.derivative(nu=2)(self.time)

        # Compute yaw angle (phi) using atan2
        self.phi_d = np.arctan2(self.dy_d, self.dx_d)

        # Compute yaw rate (dphi) using the chain rule
        dphi_d = (self.dx_d * self.ddy_d - self.dy_d * self.ddx_d)
        dphi_d[np.isnan(dphi_d)] = 0  # Handle division by zero when the velocity is zero

        # handfling the iterations for the communication
        self.iteration = 1 

        # initialization of the phi
        self.phi = 0

    def set_x(self):
        self.iteration += 1
        return self.x_d[self.iteration]

    def set_y(self):
        return self.y_d[self.iteration]

    def gen_setpoint(self):
        msg_pose = PoseStamped()
        #msg_twist = Twist() ##check for that 
        msg_pose.header.seq = self.iteration
        msg_pose.pose.position.x = self.x_d[self.iteration]
        msg_pose.pose.position.y = self.y_d[self.iteration]
        msg_pose.pose.orientation.z = self.phi_d

        ## Iteration update
        self.iteration +=1 
        rospy.loginfo("Guidance: Sending waypoit: x={:.3f},".format(msg_pose.pose.position.x))
        rospy.loginfo("waypoint number: iter={:.3f}," .format(self.iteration))
        return msg_pose 

def sawtooth(x):
    # return the modulo 2*pi of the angle x in [-pi,pi]
    return 2*np.arctan(np.tan(x/2))

def speed_feed_forward(vx):
    # convert the desired surge into a thrust (static model)
    # TODO: the model has not been identified yet, so I put a linear tranfomation
    tau_x = 10*vx
    return tau_x

class TriangleGuidancePathFollowing:
    # guide the ROV for the triangle mission (A->B->C)
    # with a path planning method
    # based on the current position, it computes a reference for speed and yaw
    # step 1 from A to B
    # step 2 turn toward C
    # step 3 from B to C
    # step 4 turn toward A
    # step 5 from C to A
    # step 6 turn toward B
    # goto step 1

    # a target position is reached when the robot is behind the target (half plane)
    # a target angle is reached when the angle between the robot and the target is less than 10 degrees

    def __init__(self, A, B, C, u0,th_yaw=10,line_follow_gain=0.5):
        self.A = A # first point of the triangle A = [xa, ya] np array
        self.B = B # second point of the triangle B = [xb, yb] np array
        self.C = C # third point of the triangle C = [xc, yc] np array
        self.state = 1 # state of the guidance automaton in {1,2,3,4,5,6}
        self.target = B # current target position
        self.previous_target = A # previous target position
        self.u0 = u0 # speed when moving to a target
        self.th_yaw = th_yaw # threshold (deg) for when the angle is reached when turning toward a target
        self.line_follow_gain = line_follow_gain # gain for the line following system
        print("state ", self.state)

    def update(self, X):
        # update the state of the guidance automaton
        # X = [x, y, yaw] np array
        if self.state == 1:
            target_reached = np.dot(self.previous_target - self.target, X[0:2] - self.target) < 0
            if target_reached:
                self.state = 2
                self.previous_target = self.target
                self.target = self.C
                print("state ", self.state)
        elif self.state == 2:
            yaw_toward_target = np.arctan2(self.target[1] - X[1], self.target[0] - X[0])
            angle_reached = np.abs(sawtooth(yaw_toward_target - X[2])) < self.th_yaw * np.pi / 180

            if angle_reached:
                self.state = 3
                print("state ", self.state)
        elif self.state == 3:
            target_reached = np.dot(self.previous_target - self.target, X[0:2] - self.target) < 0
            if target_reached:
                self.state = 4
                self.previous_target = self.target
                self.target = self.A
                print("state ", self.state)
        elif self.state == 4:
            yaw_toward_target = np.arctan2(self.target[1] - X[1], self.target[0] - X[0])
            angle_reached = np.abs(sawtooth(yaw_toward_target - X[2])) < self.th_yaw * np.pi / 180
            if angle_reached:
                self.state = 5
                print("state ", self.state)
        elif self.state == 5:
            target_reached = np.dot(self.previous_target - self.target, X[0:2] - self.target) < 0
            if target_reached:
                self.state = 6
                self.previous_target = self.target
                self.target = self.B
                print("state ", self.state)
        elif self.state == 6:
            yaw_toward_target = np.arctan2(self.target[1] - X[1], self.target[0] - X[0])
            angle_reached = np.abs(sawtooth(yaw_toward_target - X[2])) < self.th_yaw * np.pi / 180
            if angle_reached:
                self.state = 1
                print("state ", self.state)

    def guidance(self, X):
        # compute the desired yaw angle yaw_d
        # and the desired speed u_d

        if self.state % 2 == 0:
            # turning toward a target
            yaw_toward_target = np.arctan2(self.target[1] - X[1], self.target[0] - X[0])
            return yaw_toward_target, 0.

        else:
            # line guidance system
            line_vect = self.target - self.previous_target
            yaw_line = np.arctan2(line_vect[1], line_vect[0])

            # projected point on the line
            pc = self.previous_target + np.dot(X[0:2] - self.previous_target, line_vect) / (
                        np.linalg.norm(line_vect) ** 2) * (line_vect)

            distance_to_line = np.linalg.norm(X[0:2] - pc)
            if np.cross(X[0:2] - pc, line_vect) > 0:
                distance_to_line = -distance_to_line

            yaw_d = yaw_line + np.arctan(-self.line_follow_gain*distance_to_line)
            return yaw_d, self.u0

class TriangleGuidanceTrajectoryTracking:
    # guide the ROV for the triangle mission (A->B->C)
    # with a trajectory tracking method
    # based on the current time, it computes a desired pose and its derivatives (pd,pd_dot,pd_ddot)
    # step 1 from A to B
    # step 2 turn toward C
    # step 3 from B to C
    # step 4 turn toward A
    # step 5 from C to A
    # step 6 turn toward B
    # goto step 1

    # each step last for a time T/6 where T is the period of the trajectory

    def __init__(self, A, B, C,T):
        self.A = A # 1st point of the triangle
        self.B = B # 2nd point of the triangle
        self.C = C # 3rd point of the triangle
        self.T = T # period of the trajectory from A to A

    def get_reference(self,t):
        # input t : time since the beginning of the mission (s)
        # output pd, pd_dot and pd_ddot (trajectory)
        Dt = t % self.T # time in the trajectory loop
        t_step = self.T/6 # time step for each step
        Dt_step = Dt % t_step # time in the step

        # time split in 6 steps
        if Dt < t_step:
            # goto B
            yaw_target = np.arctan2(self.B[1] - self.A[1], self.B[0] - self.A[0])
            p0 = np.array([self.A[0], self.A[1], yaw_target])
            p1 = np.array([self.B[0], self.B[1], yaw_target])

        elif Dt < 2*t_step:
            # turn toward C
            old_yaw_target = np.arctan2(self.B[1] - self.A[1], self.B[0] - self.A[0])
            yaw_target = np.arctan2(self.C[1] - self.B[1], self.C[0] - self.B[0])
            p0 = np.array([self.B[0], self.B[1], old_yaw_target])
            p1 = np.array([self.B[0], self.B[1], yaw_target])
        elif Dt < 3*t_step:
            # goto C
            yaw_target = np.arctan2(self.C[1] - self.B[1], self.C[0] - self.B[0])
            p0 = np.array([self.B[0], self.B[1], yaw_target])
            p1 = np.array([self.C[0], self.C[1], yaw_target])
        elif Dt < 4*t_step:
            # turn toward A
            old_yaw_target = np.arctan2(self.C[1] - self.B[1], self.C[0] - self.B[0])
            yaw_target = np.arctan2(self.A[1] - self.C[1], self.A[0] - self.C[0])
            p0 = np.array([self.C[0], self.C[1], old_yaw_target])
            p1 = np.array([self.C[0], self.C[1], yaw_target])
        elif Dt < 5*t_step:
            # goto A
            yaw_target = np.arctan2(self.A[1] - self.C[1], self.A[0] - self.C[0])
            p0 = np.array([self.C[0], self.C[1], yaw_target])
            p1 = np.array([self.A[0], self.A[1], yaw_target])
        else:
            # turn toward B
            old_yaw_target = np.arctan2(self.A[1] - self.C[1], self.A[0] - self.C[0])
            yaw_target = np.arctan2(self.B[1] - self.A[1], self.B[0] - self.A[0])
            p0 = np.array([self.A[0], self.A[1], old_yaw_target])
            p1 = np.array([self.A[0], self.A[1], yaw_target])
        pd, pd_dot, pd_ddot = self.interpolated_trajectory(p0, p1, Dt_step/t_step)
        pd_dot = 1/t_step*pd_dot
        pd_ddot = 1/t_step**2*pd_ddot
        return pd, pd_dot, pd_ddot

    def interpolated_trajectory(self, P0, P1, x):
        # input P0, P1 : 2D poses [x,y,yaw]
        # input x : normalized step time [0,1]
        # trajectory interpolation from P0 to P1
        # must go from pose P0 to pose P1
        w,w_dot,w_ddot = self.function_w(x) # interpolation variables
        dp = P1-P0
        dp[2] = sawtooth(dp[2])
        xy = P0 + w*dp
        xy_dot = w_dot*dp
        xy_ddot = w_ddot*dp
        return xy, xy_dot, xy_ddot

    def function_w(self,x):
        # x in [0,1]
        # smooth transition from 0 to 1
        w = - 1/(np.pi*2) * np.sin(2*np.pi*x) + x
        w_dot = 1-np.cos(2*np.pi*x)
        w_ddot = 2*np.pi*np.sin(2*np.pi*x)

        # alternative, linear motion
        # w = x
        # w_dot = 1
        # w_ddot = 0
        return w,w_dot,w_ddot