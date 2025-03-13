#!/usr/bin/env python3
import rospy
import numpy as np
import csv 
import os
from std_msgs.msg import Float64 ## Ceto
#from sensor_msgs.msg import NavSatFix ## Galene
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from thruster_manager_v2 import ThrusterManager  
from create_new_csv_file import create_new_csv_file
from geometry_msgs.msg import PoseStamped
from guidance import Guidance
from phe import paillier # for Paillier cryptography 
import time

# Configuration parameters
USE_ENCRYPTION = True  # Master switch for encryption
ENCRYPTION_FREQUENCY = 10  # Hz - how often to run encrypted calculations
ENCRYPT_CRITICAL_ONLY = True  # Only encrypt attitude (roll/pitch) data if True
PUBLIC_KEY_SIZE = 256  # Smaller key size for better performance (512, 768, 1024, or 3072)

# Generate keypair with specified key size for better performance
global P, S
if USE_ENCRYPTION:
    P, S = paillier.generate_paillier_keypair(n_length=PUBLIC_KEY_SIZE)
    rospy.loginfo(f"Generated Paillier keypair with {PUBLIC_KEY_SIZE}-bit key")
else:
    # Define P and S as None if not using encryption
    P, S = None, None

############################### PID Class #################################
class PID:
    """Full PID controller implementation with homomorphic encryption support."""
    def __init__(self, kp, ki, kd, setpoint=None, integral_limit=4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit  # Limit for integral term
        
        # Timing statistics
        self.total_time = 0
        self.call_count = 0
        self.max_time = 0
        self.min_time = float('inf')
        
        # Handle encryption properly for initial values
        if USE_ENCRYPTION:
            if isinstance(setpoint, paillier.EncryptedNumber):
                self.setpoint = setpoint
            elif setpoint is None:
                self.setpoint = P.encrypt(0)
            else:
                self.setpoint = P.encrypt(setpoint)
                
            # Initialize with encrypted zeros
            self.prev_error = P.encrypt(0)
            self.integral = P.encrypt(0)
        else:
            self.setpoint = 0 if setpoint is None else setpoint
            self.prev_error = 0
            self.integral = 0
            
    def update(self, measurement, setpoint=None):
        """Calculate the PID control output with support for encrypted values."""
        start_time = time.time()
        rospy.loginfo(f"PID update start time: {start_time}")
        
        if setpoint is None:  # Use the default setpoint if not provided
            setpoint = self.setpoint
            
        # Calculate error terms
        error = setpoint - measurement
        rospy.loginfo("Error check")
        # Update integral term
        self.integral = self.integral + error
        rospy.loginfo("Integral check")
        # Limit integral term if not encrypted
        # Cannot clip encrypted values directly - would need secure comparison
        if not USE_ENCRYPTION and self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
            
        # Calculate derivative term
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Compute PID output
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        result = p_term + i_term + d_term
        rospy.loginfo("PID check")
        # Calculate timing
        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.call_count += 1
        self.max_time = max(self.max_time, elapsed)
        self.min_time = min(self.min_time, elapsed)
        
        # Print timing info every 10 calls
        if self.call_count % 10 == 0:
            avg_time = self.total_time / self.call_count
            rospy.loginfo(f"PID timing - Avg: {avg_time*1000:.2f}ms, Min: {self.min_time*1000:.2f}ms, Max: {self.max_time*1000:.2f}ms, Count: {self.call_count}")
        
        return result
    
############################ ROV_PID_Controller Class ########################################
class ROVPIDController:
    """PID-based controller for an ROV using ROS with homomorphic encryption support."""
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

        # Initialize sensor values
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.altitude = 0
        self.x = 0
        self.y = 0
        self.time_stamp = None
        
        # Initialize encrypted values
        if USE_ENCRYPTION:
            self.safe_roll = None
            self.safe_pitch = None
            self.safe_yaw = None
            self.safe_altitude = None
            self.safe_x = None
            self.safe_y = None
            self.safe_set_x = None
            self.safe_set_y = None
            self.safe_set_altitude = None
            self.safe_set_roll = None
            self.safe_set_pitch = None
            self.safe_set_yaw = None

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
        # Initialize PID controllers with full PID terms
        if USE_ENCRYPTION:
            self.pid_yaw = PID(kp=0.03, ki=0.01, kd=0.04, setpoint=P.encrypt(170))
            self.pid_z = PID(kp=4.0, ki=0.4, kd=1.8, setpoint=P.encrypt(17.1))
            self.pid_x = PID(kp=12.0, ki=0.004, kd=0.3, setpoint=P.encrypt(0))
            self.pid_y = PID(kp=2.45, ki=0.005, kd=0.008, setpoint=P.encrypt(0))
            self.pid_roll = PID(kp=0.02, ki=0.01, kd=0.18, setpoint=P.encrypt(0))  
            self.pid_pitch = PID(kp=0.046, ki=0.01, kd=0.14, setpoint=P.encrypt(0))
        else:
            self.pid_yaw = PID(kp=0.03, ki=0.01, kd=0.04, setpoint=170)
            self.pid_z = PID(kp=4.0, ki=0.4, kd=1.8, setpoint=17.1)
            self.pid_x = PID(kp=12.0, ki=0.004, kd=0.3, setpoint=0)
            self.pid_y = PID(kp=2.45, ki=0.005, kd=0.008, setpoint=0)
            self.pid_roll = PID(kp=0.02, ki=0.01, kd=0.18, setpoint=0)  
            self.pid_pitch = PID(kp=0.046, ki=0.01, kd=0.14, setpoint=0)

        # CSV Logging
        self.csv_file = create_new_csv_file()
        self.init_csv()

        rospy.sleep(5) ## wait for the guidance to be ready

        ## Generate the trajectory (define the function generate_trajectory)
        self.Trajectory = Guidance(self.x0, self.y0)

        self.control_loop()  # Start control loop

##################################### callback functions ########################################
    def imu_callback(self, msg):
        """Update roll, pitch, and yaw values from IMU data."""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(qx, qy, qz, qw)
        
        # Encrypt values if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_roll = P.encrypt(self.roll)
                self.safe_pitch = P.encrypt(self.pitch)
                self.safe_yaw = P.encrypt(self.yaw)
            except Exception as e:
                rospy.logerr(f"Error encrypting IMU data: {e}")

    def altitude_callback(self, msg):
        """Update altitude value."""
        self.altitude = msg.data
        
        # Encrypt altitude if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_altitude = P.encrypt(self.altitude)
            except Exception as e:
                rospy.logerr(f"Error encrypting altitude data: {e}")

    def pose_callback(self, msg):
        """Update x and y values from the Optitrack."""
        ## The orientation of the global frame is different from the local frame
        self.x = -msg.pose.position.x
        self.y = msg.pose.position.z 
        self.time_stamp = msg.header.stamp
        
        # Encrypt position if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_x = P.encrypt(self.x)
                self.safe_y = P.encrypt(self.y)
            except Exception as e:
                rospy.logerr(f"Error encrypting position data: {e}")
                
        #rospy.loginfo("New Pose received: x={:.3f}, y={:.3f}".format(self.x, self.y))

    def quaternion_to_euler(self, x, y, altitude, w):
        """Convert quaternion to Euler angles."""
        r = R.from_quat([x, y, altitude, w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return roll, pitch, yaw

##################################### csv functions ########################################
    def init_csv(self):
        """Initialize the CSV file if it does not exist, including PID gain information."""
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='a', newline='') as file:
                # Write PID gain information as header comments
                file.write("# PID Gains:\n")
                file.write("# Roll:  kp={}, ki={}, kd={}\n".format(self.pid_roll.kp, self.pid_roll.ki, self.pid_roll.kd))
                file.write("# Pitch: kp={}, ki={}, kd={}\n".format(self.pid_pitch.kp, self.pid_pitch.ki, self.pid_pitch.kd))
                file.write("# Yaw:   kp={}, ki={}, kd={}\n".format(self.pid_yaw.kp, self.pid_yaw.ki, self.pid_yaw.kd))
                file.write("# Z:     kp={}, ki={}, kd={}\n".format(self.pid_z.kp, self.pid_z.ki, self.pid_z.kd))
                file.write("# X:     kp={}, ki={}, kd={}\n".format(self.pid_x.kp, self.pid_x.ki, self.pid_x.kd))
                file.write("# Y:     kp={}, ki={}, kd={}\n".format(self.pid_y.kp, self.pid_y.ki, self.pid_y.kd))
                file.write("# x0={}, y0={}, z0={}\n".format(self.x0, self.y0, self.z0))
                file.write("# roll0={}, pitch0={}, yaw0={}\n".format(self.roll0, self.pitch0, self.yaw0))
                file.write("# Starting time: {}\n".format(self.starting_time))
                
                # Write encryption info
                file.write("# Encryption Configuration:\n")
                file.write("# USE_ENCRYPTION: {}\n".format(USE_ENCRYPTION))
                if USE_ENCRYPTION:
                    file.write("# ENCRYPTION_FREQUENCY: {} Hz\n".format(ENCRYPTION_FREQUENCY))
                    file.write("# KEY_SIZE: {} bits\n".format(PUBLIC_KEY_SIZE))
                    file.write("# ENCRYPT_CRITICAL_ONLY: {}\n".format(ENCRYPT_CRITICAL_ONLY))
                
                # Write header row for the CSV data
                writer = csv.writer(file)
                writer.writerow(["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", 
                                 "Pitch", "Roll", "Yaw", "Altitude",
                                 "x", "y", "set_x", "set_y", "set_altitude", "set_roll", "set_pitch", "set_yaw",
                                 "Thruster_1", "Thruster_2", "Thruster_3", 
                                 "Thruster_4", "Thruster_5", "Thruster_6",
                                 "Encrypted"])

    def log_to_csv(self, Fx, Fy, Fz, Mx, My, Mz, pitch, roll, yaw, altitude, x, y, set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw, thrust_forces, is_encrypted=False):
        """Log data to CSV."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rospy.get_time(), Fx, Fy, Fz, Mx, My, Mz, pitch, roll, yaw, altitude, x, y, set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw] 
                           + list(thrust_forces) + [1 if is_encrypted else 0])

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
        # Initialize with current values first as a fallback
        self.x0 = self.x
        self.y0 = self.y
        self.pitch0 = self.pitch
        self.roll0 = self.roll
        self.z0 = self.altitude
        self.yaw0 = self.yaw
        self.starting_time = rospy.Time.now().to_sec()
        
        stable = False
        max_attempts = 3
        attempt = 0
    
        while not stable and attempt < max_attempts and not rospy.is_shutdown():
            # Store current position and orientation
            possible_x = self.x
            possible_y = self.y
            possible_pitch = self.pitch
            possible_roll = self.roll
            possible_z = self.altitude 
            possible_yaw = self.yaw 
        
            # Wait for stability
            rospy.loginfo(f"Waiting for stability check (attempt {attempt+1}/{max_attempts})...")
            rospy.sleep(2)
        
            # Check if position and orientation are still the same
            pos_stable = (abs(possible_x - self.x) < 0.05 and 
                        abs(possible_y - self.y) < 0.05 and 
                        abs(possible_z - self.altitude) < 0.05)
        
            # Use larger threshold for orientation (in degrees)
            ori_stable = (abs(possible_pitch - self.pitch) < 1.0 and 
                        abs(possible_roll - self.roll) < 1.0 and 
                        abs(possible_yaw - self.yaw) < 1.0) 
            
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
                            f"x={self.x0:.3f}, y={self.y0:.3f}, altitude={self.z0:.3f}, "
                            f"pitch={self.pitch0:.2f}°, roll={self.roll0:.2f}°, yaw={self.yaw0:.2f}°")
            else:
                attempt += 1
                rospy.logwarn(f"Not stable. Position stable: {pos_stable}, Orientation stable: {ori_stable}")
                rospy.logwarn(f"Current: (x={self.x:.3f}, y={self.y:.3f}, pitch={self.pitch:.2f}°, roll={self.roll:.2f}°)")
                rospy.logwarn(f"Previous: (x={possible_x:.3f}, y={possible_y:.3f}, "
                            f"pitch={possible_pitch:.2f}°, roll={possible_roll:.2f}°)")
        
        if not stable:
            rospy.logwarn("Could not establish stability. Using current values as starting point.")

################################# control loop ##############################################
    def control_loop(self):
        """Main control loop (10 Hz) with lower frequency encryption."""
        rate = rospy.Rate(10)  # 10 Hz for control loop
        start_time = rospy.Time.now().to_sec()
        
        # Initialize encryption timer
        encryption_cycle_time = 1.0 / ENCRYPTION_FREQUENCY if ENCRYPTION_FREQUENCY > 0 else float('inf')
        last_encryption_time = start_time
        
        # Cache for encrypted control outputs
        cached_Fx, cached_Fy, cached_Fz = 0, 0, 0
        cached_Mx, cached_My, cached_Mz = 0, 0, 0
        
        # Log encryption configuration
        if USE_ENCRYPTION:
            rospy.loginfo(f"Using full PID with encryption: {PUBLIC_KEY_SIZE}-bit key at {ENCRYPTION_FREQUENCY}Hz")
            if ENCRYPT_CRITICAL_ONLY:
                rospy.loginfo("Encrypting critical attitude data only")
        else:
            rospy.loginfo("Using full PID control without encryption")
        
        while not rospy.is_shutdown():
            try:
                # Get desired location from trajectory
                self.set_x = self.Trajectory.set_x()
                self.set_y = self.Trajectory.set_y()
                self.set_altitude = self.z0
                self.set_roll = self.roll0
                self.set_pitch = self.pitch0
                self.set_yaw = self.yaw0
                
                current_time = rospy.Time.now().to_sec()
                
                # Decide whether to use encryption this cycle
                use_encryption_this_cycle = (USE_ENCRYPTION and 
                                           (current_time - last_encryption_time >= encryption_cycle_time))
                
                if use_encryption_this_cycle:
                    # Reset timer for next encryption cycle
                    last_encryption_time = current_time
                    rospy.loginfo("Running encrypted control cycle")
                    
                    try:
                        # Encrypt setpoints
                        if USE_ENCRYPTION:
                            self.safe_set_x = P.encrypt(self.set_x)
                            self.safe_set_y = P.encrypt(self.set_y)
                            self.safe_set_altitude = P.encrypt(self.set_altitude)
                            self.safe_set_roll = P.encrypt(self.set_roll)
                            self.safe_set_pitch = P.encrypt(self.set_pitch)
                            self.safe_set_yaw = P.encrypt(self.set_yaw)
                        
                        # Only encrypt what we need
                        if ENCRYPT_CRITICAL_ONLY:
                            # Only encrypt critical attitude data (roll/pitch)
                            # Compute encrypted attitude control
                            Mx_safe = self.pid_roll.update(self.safe_roll, self.safe_set_roll)
                            My_safe = self.pid_pitch.update(self.safe_pitch, self.safe_set_pitch)
                            
                            # Non-encrypted calculations for others
                            Mz = self.pid_yaw.update(self.yaw, self.set_yaw)
                            Fz = self.pid_z.update(self.altitude, self.set_altitude)
                            Fx = self.pid_x.update(self.x, self.set_x)
                            Fy = self.pid_y.update(self.y, self.set_y)
                            
                            # Decrypt only what was encrypted
                            Mx = S.decrypt(Mx_safe)
                            My = S.decrypt(My_safe)
                        else:
                            # Compute all encrypted PID corrections
                            Mx_safe = self.pid_roll.update(self.safe_roll, self.safe_set_roll)
                            My_safe = self.pid_pitch.update(self.safe_pitch, self.safe_set_pitch)
                            Mz_safe = self.pid_yaw.update(self.safe_yaw, self.safe_set_yaw)
                            Fz_safe = self.pid_z.update(self.safe_altitude, self.safe_set_altitude)
                            Fx_safe = self.pid_x.update(self.safe_x, self.safe_set_x)
                            Fy_safe = self.pid_y.update(self.safe_y, self.safe_set_y)
                            
                            # Decrypt all PID outputs
                            Mx = S.decrypt(Mx_safe)
                            My = S.decrypt(My_safe)
                            Mz = S.decrypt(Mz_safe)
                            Fz = S.decrypt(Fz_safe)
                            Fx = S.decrypt(Fx_safe)
                            Fy = S.decrypt(Fy_safe)
                        
                        # Cache the results for future unencrypted cycles
                        cached_Fx, cached_Fy, cached_Fz = Fx, Fy, Fz
                        cached_Mx, cached_My, cached_Mz = Mx, My, Mz
                        
                    except Exception as e:
                        rospy.logerr(f"Error in encrypted PID computation: {e}")
                        # Fall back to non-encrypted computation
                        Mx = self.pid_roll.update(self.roll, self.set_roll)
                        My = self.pid_pitch.update(self.pitch, self.set_pitch)
                        Mz = self.pid_yaw.update(self.yaw, self.set_yaw)
                        Fz = self.pid_z.update(self.altitude, self.set_altitude)
                        Fx = self.pid_x.update(self.x, self.set_x)
                        Fy = self.pid_y.update(self.y, self.set_y)
                        
                        # Update cache with non-encrypted values
                        cached_Fx, cached_Fy, cached_Fz = Fx, Fy, Fz
                        cached_Mx, cached_My, cached_Mz = Mx, My, Mz
                else:
                    if USE_ENCRYPTION:
                        # Use cached values from previous encrypted cycle
                        Fx, Fy, Fz = cached_Fx, cached_Fy, cached_Fz
                        Mx, My, Mz = cached_Mx, cached_My, cached_Mz
                    else:
                        # Use regular non-encrypted PID
                        Mx = self.pid_roll.update(self.roll, self.set_roll)
                        My = self.pid_pitch.update(self.pitch, self.set_pitch)
                        Mz = self.pid_yaw.update(self.yaw, self.set_yaw)
                        Fz = self.pid_z.update(self.altitude, self.set_altitude)
                        Fx = self.pid_x.update(self.x, self.set_x)
                        Fy = self.pid_y.update(self.y, self.set_y)

                # Log current sensor data
                #rospy.loginfo("IMU Data: Roll={:.2f}°, Pitch={:.2f}°, Yaw={:.2f}°".format(self.roll, self.pitch, self.yaw))
                #rospy.loginfo("altitude: {:.2f} m".format(self.altitude))
                #rospy.loginfo("Pose: x={:.3f}, y={:.3f}".format(self.x, self.y))
                #rospy.loginfo("PID Outputs: Fx={:.3f}, Fy={:.3f}, Fz={:.3f}, Mx={:.3f}, My={:.3f}, Mz={:.3f}".format(Fx, Fy, Fz, Mx, My, Mz))

                # Create control vector
                tau = np.array([Fx*0, Fy*0, Fz*0, Mx, My, Mz])
                rospy.loginfo(f"Computed Control: Fx={Fx:.3f}, Fy={Fy:.3f}, Fz={Fz:.3f}, Mx={Mx:.3f}, My={My:.3f}, Mz={Mz:.3f}")

                # Compute thrust forces
                thrust_forces = np.linalg.pinv(self.tcm_enu) @ tau
                thrust_forces = np.clip(thrust_forces, -20, 25)
                rospy.loginfo(f"Thruster Forces: {np.round(thrust_forces, 4)}")

                # Send thruster commands
                self.thruster_manager.send_thruster_commands(thrust_forces, Delta_T=0.01)

                # Log data to CSV - include encryption status
                self.log_to_csv(Fx, Fy, Fz, Mx, My, Mz, self.pitch, self.roll, self.yaw, self.altitude, 
                              self.x, self.y, self.set_x, self.set_y, self.set_altitude, 
                              self.set_roll, self.set_pitch, self.set_yaw, thrust_forces,
                              is_encrypted=use_encryption_this_cycle)
                
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
                
            rate.sleep()


if __name__ == "__main__":
    try:
        ROVPIDController()
    except rospy.ROSInterruptException:
        pass