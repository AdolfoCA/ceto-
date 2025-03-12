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


