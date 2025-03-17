#!/usr/bin/env python3
import rospy
import numpy as np
import csv 
import os
from std_msgs.msg import Float64 ## Ceto
from sensor_msgs.msg import NavSatFix ## Galene
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

from scripts.guidance import TriangleGuidancePathFollowing, speed_feed_forward,TriangleGuidanceTrajectoryTracking
from thruster_manager_v2 import ThrusterManager  
from create_new_csv_file import create_new_csv_file
from geometry_msgs.msg import PoseStamped
from guidance import Guidance
from direct_rc import override_rc


############################### PID Class #################################
class PID:
    """Simple PID controller implementation."""
    def __init__(self, kp, ki, kd, setpoint=0., integral_limit=4.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = integral_limit  

    def update(self, measurement,setpoint=None):
        """Calculate the PID control output."""
        if setpoint == None: # Use the default setpoint if not provided
            setpoint = self.setpoint
        # Calculate error terms
        error = setpoint - measurement
        self.integral += error
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)  
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative 
############################### ROV PID Controller #################################
class ROVPIDController:
    """PID-based controller for an ROV using ROS."""
    def __init__(self):
        rospy.init_node("rov_pid_controller", anonymous=True)

        self.start_time = rospy.Time.now().to_sec()

        # Initialize Thruster Manager
        self.thruster_manager = ThrusterManager()

        # Ensure the vehicle is armed before starting the PID control
        if not self.thruster_manager.current_state or not self.thruster_manager.current_state.armed:
            rospy.loginfo("Arming vehicle before starting PID control...")
            self.thruster_manager.arm_vehicle()

        # Set MANUAL mode
        rospy.loginfo("Setting MANUAL mode...")
        self.thruster_manager.set_manual_mode()

        # Thruster Configuration Matrix (ENU)
        self.tcm_enu = np.array([
            [ 1.0,     1.0,     0.0,     0.0,     0.0,      0.0   ],
            [ 0.0,     0.0,     0.0,     0.0,     0.0,      1.0   ],
            [ 0.0,     0.0,     1.0,     1.0,     1.0,      0.0   ],
            [ 0.0,    -0.0,    -0.1105,  0.1105,  0.0,     -0.098 ],
            [-0.0105, -0.0105, -0.1014, -0.1014,  0.2502,   0.0   ],
            [ 0.1105, -0.1105,  0.0,     0.0,    -0.0,     -0.037 ]
        ])

        # Subscribe to IMU and altitude data
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        #self.z_sub = rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.altitude_callback) ## Galene
        self.altitude_sub = rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.altitude_callback) ## Ceto


        # Subscribe to Optitrack pose data
        rospy.loginfo("Subscribing to /vrpn_client_node/Galene/pose.")
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/Galene/pose", PoseStamped, self.pose_callback)

        # Register a shutdown callback to disarm the vehicle on exit
        rospy.on_shutdown(self.shutdown_callback)

        rospy.loginfo("Waiting for IMU and altitude data...")
        rospy.sleep(2)  # Wait for the first data samples

        # Initialize variables 
        self.variable_init()

        self.roll = self.roll0
        self.pitch = self.pitch0
        self.yaw = self.yaw0
        self.altitude = self.z0
        self.x = self.x0
        self.y = self.y0

        ## Now that i have checked the equilibrium for the roll and pitch --> set that as setpoint in the PID 
        # Initialize PID controllers 
        self.pid_yaw = PID(kp=0.03, ki=0.01, kd=0.04, setpoint=170)
        self.pid_z = PID(kp=4.0, ki=0.4, kd=1.8, setpoint=17.1)
        # self.pid_x = PID(kp=12.0, ki=0.004, kd=0.3, setpoint=0)
        # self.pid_y = PID(kp=2.45, ki=0.005, kd=0.008, setpoint=0)

        self.pid_roll = PID(kp=0.02, ki=0.01, kd=0.18, setpoint=0)  
        self.pid_pitch = PID(kp=0.046, ki=0.01, kd=0.14, setpoint=0)

        # CSV Logging
        self.csv_file = create_new_csv_file()
        self.init_csv()

        rospy.sleep(5) ## wait for the guidance to be ready

        ## Generate the trajectory (define the function generate_trajectory)
        # self.Trajectory = Guidance(self.x0,self.y0)


        u0 = 1 # speed when moving to a target
        A = np.array([self.x0,self.y0])
        B = np.array([0.,0.])
        C = np.array([1.,0.])
        self.Guidance = TriangleGuidancePathFollowing(A, B, C, u0,th_yaw=10,line_follow_gain=0.5)

        # alternative guidance
        # T_trajectory = 60 # time to complete the trajectory (s)
        # self.Guidance2 =  TriangleGuidanceTrajectoryTracking(A, B, C, T_trajectory)

        self.control_loop()  # Start control loop


##################################### callback functions ########################################

    def imu_callback(self, msg):
        """Update roll, pitch, and yaw values from IMU data."""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(qx, qy, qz, qw)

    def altitude_callback(self, msg):
        """Update altitude value."""
        self.altitude = msg.data

    def pose_callback(self, msg):
        """Update x and y values from the Optitrack."""
        ## The orientation of the global frame is different from the local frame
        self.x = - msg.pose.position.x
        self.y = msg.pose.position.z 
        self.time_stamp = msg.header.stamp
        rospy.loginfo("New Pose received: x={:.3f}, y={:.3f}".format(self.x, self.y))


    def quaternion_to_euler(self, x, y, altitude, w):
        """Convert quaternion to Euler angles."""
        r = R.from_quat([x, y, altitude, w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return roll, pitch, yaw

##################################### cvs functions ########################################

    def init_csv(self):
        """Initialize the CSV file if it does not exist, including PID gain information."""
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='a', newline='') as file:
                # Write PID gain information as header comments
                file.write("# PID Gains:\n")
                file.write("# Roll:  kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_roll.kp, self.pid_roll.ki, self.pid_roll.kd, self.pid_roll.setpoint))
                file.write("# Pitch: kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_pitch.kp, self.pid_pitch.ki, self.pid_pitch.kd, self.pid_pitch.setpoint))
                file.write("# Yaw:   kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_yaw.kp, self.pid_yaw.ki, self.pid_yaw.kd, self.pid_yaw.setpoint))
                file.write("# Z:     kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_z.kp, self.pid_z.ki, self.pid_z.kd, self.pid_z.setpoint))
                file.write("# X:     kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_x.kp, self.pid_x.ki, self.pid_x.kd, self.pid_x.setpoint))
                file.write("# Y:     kp={}, ki={}, kd={}, setpoint={}\n".format(self.pid_y.kp, self.pid_y.ki, self.pid_y.kd, self.pid_y.setpoint))
                file.write("# x0={}, y0 = {}, z0={}\n".format(self.x0, self.y0,self.z0))
                file.write("# roll0={}, pitch0={}, yaw0={}\n".format(self.roll0, self.pitch0, self.yaw0))


                file.write("# Starting time: {}\n".format(self.starting_time))
                # Write header row for the CSV data
                writer = csv.writer(file)
                writer.writerow(["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", 
                                 "Pitch", "Roll", "Yaw", "Altitude",
                                 "x", "y", "set_x", "set_y", "set_altitude", "set_roll", "set_pitch", "set_yaw",
                                 "Thruster_1", "Thruster_2", "Thruster_3", 
                                 "Thruster_4", "Thruster_5", "Thruster_6"])

    def log_to_csv(self, Fx, Fy, Fz, Mx, My, Mz, pitch, roll, yaw, altitude, x, y, set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw, thrust_forces):
        """Log data to CSV."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rospy.get_time(), Fx, Fy, Fz, Mx, My, Mz, pitch, roll, yaw, altitude, x, y, set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw] + list(thrust_forces))

    def shutdown_callback(self):
        """Disarm the vehicle when the node is shutting down."""
        rospy.loginfo("Shutting down... Disarming vehicle.")
        self.thruster_manager.disarm_vehicle()

##################################### initialization functions ########################################
    def variable_init(self):
        """
        Initialize starting position and orientation by checking if values are stable.
        If position and orientation remain within threshold after 5 seconds, set as starting values.
        """
        stable = False
        max_attempts = 3
        attempt = 0
    
        while not stable and attempt < max_attempts and not rospy.is_shutdown():
        # Store current position and orientation
            self.possible_x = self.x
            self.possible_y = self.y
            self.possible_pitch = self.pitch
            self.possible_roll = self.roll
            self.possible_z = self.altitude 
            self.possible_yaw = self.yaw 
        
        # Wait for 5 seconds
            rospy.loginfo(f"Waiting for stability check (attempt {attempt+1}/{max_attempts})...")
            rospy.sleep(2)
        
        # Check if position and orientation are still the same
            pos_stable = (abs(self.possible_x - self.x) < 0.05 and 
                        abs(self.possible_y - self.y) < 0.05 and 
                        abs (self.possible_z - self.altitude) < 0.05)
        
            # Use larger threshold for orientation (in degrees)
            ori_stable = (abs(self.possible_pitch - self.pitch) < 1.0 and 
                        abs(self.possible_roll - self.roll) < 1.0 and 
                        abs (self.possible_yaw - self.yaw) < 1.0 ) 
            
            if pos_stable and ori_stable:
                stable = True
                self.x0 = self.x
                self.y0 = self.y
                self.pitch0 = self.pitch
                self.roll0 = self.roll
                self.z0 = self.altitude 
                self.yaw0 = self.yaw 
                self.starting_time = rospy.Time.now().to_sec()
                rospy.loginfo(f"Position and orientation stable. Starting values: "
                            f"x={self.x0:.3f}, y={self.y0:.3f}, altitude = {self.z0:.3f}"
                            f"pitch={self.pitch0:.2f}°, roll={self.roll0:.2f}°, yaw={self.yaw0:.2f}°")
            else:
                attempt += 1
                rospy.logwarn(f"Not stable. Position stable: {pos_stable}, Orientation stable: {ori_stable}")
                rospy.logwarn(f"Current: (x={self.x:.3f}, y={self.y:.3f}, pitch={self.pitch:.2f}°, roll={self.roll:.2f}°)")
                rospy.logwarn(f"Previous: (x={self.possible_x:.3f}, y={self.possible_y:.3f}, "
                            f"pitch={self.possible_pitch:.2f}°, roll={self.possible_roll:.2f}°)")
        
        if not stable:
            rospy.logwarn("Could not establish stability. Using current values as starting point.")
            self.x0 = self.x
            self.y0 = self.y
            self.pitch0 = self.pitch
            self.roll0 = self.roll
            self.yaw0 = self.yaw 
            self.z0 = self.altitude 
            self.starting_time = rospy.Time.now().to_sec()


################################# control loop ##############################################
    def control_loop(self):
        """Main control loop (10 Hz)."""
        rate = rospy.Rate(10)  # 10 Hz
        start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():

            # update the guidance
            X =  np.array([self.x,self.y,self.yaw])
            self.Guidance.update(X)
            self.set_yaw,self.set_vx = self.Guidance.guidance(X) # reference for the yaw and the x-speed

            # alternative guidance
            # mission_time = rospy.Time.now().to_sec() - start_time
            # pd,pd_dot,pd_ddot = self.Guidance2.get_reference(mission_time)   # careful ! these are in the world frame
            # self.set_x = pd[0]
            # self.set_y = pd[1]
            # self.set_yaw = pd[2]
            # alternative guidance
            # TODO add the speed measurement and speed controller, or even the backstepping controller


            # additional set_points
            # self.set_x = self.Trajectory.set_x()
            # self.set_y = self.Trajectory.set_y()
            self.set_altitude = self.z0
            self.set_roll = self.roll0
            self.set_pitch = self.pitch0
            # self.set_yaw = self.yaw0


            # Compute PID corrections
            Mx = self.pid_roll.update(self.roll,self.set_roll)
            My = self.pid_pitch.update(self.pitch,self.set_pitch)
            Mz = self.pid_yaw.update(self.yaw,self.set_yaw)
            Fz = self.pid_z.update(self.altitude,self.set_altitude)
            # Fx = self.pid_x.update(self.x,self.set_x)
            Fx = speed_feed_forward(self.set_vx) # TODO this feedforward is arbitrary (need to identify relation bwm speed and thrust of measure speed)
            # Fy = self.pid_y.update(self.y,self.set_y)
            Fy = 0.
            #rospy.loginfo(f"Current Orientation: Roll={self.roll:.2f}°, Pitch={self.pitch:.2f}°, Yaw={self.yaw:.2f}°, altitude={self.altitude:.2f}m,  Pose: x={self.x.3f}, y={self.y.3f}")

            # Log current sensor data
            rospy.loginfo("IMU Data: Roll={:.2f}°, Pitch={:.2f}°, Yaw={:.2f}°".format(self.roll, self.pitch, self.yaw))
            rospy.loginfo("altitude: {:.2f} m".format(self.altitude))
            rospy.loginfo("Pose: x={:.3f}, y={:.3f}".format(self.x, self.y))
            rospy.loginfo("PID Outputs: Fx={:.3f}, Fy={:.3f}, Fz={:.3f}, Mx={:.3f}, My={:.3f}, Mz={:.3f}".format(Fx, Fy, Fz, Mx, My, Mz))

            # Create control vector
            tau = np.array([Fx*0, Fy*0, Fz*0, Mx, My, Mz])
            rospy.loginfo(f"Computed Control: Fx={Fx:.3f}, Fy={Fy:.3f}, Fz={Fz:.3f}, Mx={Mx:.3f}, My={My:.3f}, Mz={Mz:.3f}")

            # Compute thrust forces
            thrust_forces = np.linalg.pinv(self.tcm_enu) @ tau
            thrust_forces = np.clip(thrust_forces, -20, 25)
            rospy.loginfo(f"Thruster Forces: {np.round(thrust_forces, 4)}")

            # Send thruster commands
            self.thruster_manager.send_thruster_commands(thrust_forces, Delta_T=0.01)

            # Log data to CSV
            self.log_to_csv(Fx, Fy, Fz, Mx, My, Mz, self.pitch, self.roll, self.yaw, self.altitude, self.x, self.y, self.set_x, self.set_y, self.set_altitude, self.set_roll, self.set_pitch, self.set_yaw, thrust_forces)
            #self.log_to_csv(Fx, Fy, Fz, Mx, My, Mz, self.pitch, self.roll, self.yaw, self.altitude, thrust_forces)

            rate.sleep()


if __name__ == "__main__":
    try:
        ## To clean the queue of the motors 
        #override_rc()
        ROVPIDController()
    except rospy.ROSInterruptException:
        pass
