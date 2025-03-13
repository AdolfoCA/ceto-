#!/usr/bin/env python3
import rospy
import numpy as np
import csv 
import os
import time  # Added time module for computation time tracking
from std_msgs.msg import Float64 ## Ceto
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from thruster_manager_v2 import ThrusterManager  
from create_new_csv_file import create_new_csv_file
from geometry_msgs.msg import PoseStamped
from guidance import Guidance
from phe import paillier # for Paillier cryptography 


###
#### It is like test2 but i can save also the time of computation in the csv file
###
# Configuration parameters
USE_ENCRYPTION = True  # Master switch for encryption
ENCRYPTION_FREQUENCY = 1  # Hz - how often to run encrypted calculations (1Hz = once per second)
ENCRYPT_CRITICAL_ONLY = False  # Only encrypt attitude (roll/pitch) data if True
PUBLIC_KEY_SIZE = 128  # Smaller key size for better performance (64, 128, 256,512, 768, 1024, or 3072)

# Only generate keypair if encryption is enabled
if USE_ENCRYPTION:
    # Generate keypair with specified key size for better performance
    public_key, private_key = paillier.generate_paillier_keypair(n_length=PUBLIC_KEY_SIZE)
    rospy.loginfo(f"Generated Paillier keypair with {PUBLIC_KEY_SIZE}-bit key")
else:
    public_key, private_key = None, None

############################### PID Class #################################
class PID:
    """Simple PID controller implementation that can work with both normal and encrypted values."""
    def __init__(self, kp, setpoint=0):
        self.kp = kp
        # If the setpoint is already encrypted, use it directly
        if USE_ENCRYPTION and not isinstance(setpoint, paillier.EncryptedNumber):
            self.setpoint = public_key.encrypt(setpoint)
        else:
            self.setpoint = setpoint

    def update(self, measurement, setpoint=None):
        """Calculate the PID control output."""
        # Use the provided setpoint or fall back to the default
        if setpoint is None:
            setpoint = self.setpoint
            
        # Calculate error terms
        error = setpoint - measurement
        
        # Apply proportional control
        output = self.kp * error
        
        return output

############################ ROV_PID_Controller Class ########################################
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
        self.altitude_sub = rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.altitude_callback)

        # Subscribe to Optitrack pose data
        rospy.loginfo("Subscribing to /vrpn_client_node/Galene/pose.")
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/Galene/pose", PoseStamped, self.pose_callback)

        # Register a shutdown callback to disarm the vehicle on exit
        rospy.on_shutdown(self.shutdown_callback)

        # Initialize sensor data variables
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.altitude = 0
        self.x = 0
        self.y = 0
        self.time_stamp = None
        
        # Initialize encrypted values if using encryption
        if USE_ENCRYPTION:
            self.safe_roll = None
            self.safe_pitch = None
            self.safe_yaw = None
            self.safe_altitude = None
            self.safe_x = None
            self.safe_y = None

        rospy.loginfo("Waiting for IMU and altitude data...")
        rospy.sleep(2)  # Wait for the first data samples

        # Initialize variables 
        self.variable_init()

        # Set current values to initial values
        self.roll = self.roll0
        self.pitch = self.pitch0
        self.yaw = self.yaw0
        self.altitude = self.z0
        self.x = self.x0
        self.y = self.y0
        
        # Initialize PID controllers 
        if USE_ENCRYPTION:
            self.pid_yaw = PID(kp=0.03, setpoint=170)
            self.pid_z = PID(kp=4.0, setpoint=17.1)
            self.pid_x = PID(kp=12.0, setpoint=0)
            self.pid_y = PID(kp=2.45, setpoint=0)
            self.pid_roll = PID(kp=0.02, setpoint=0)  
            self.pid_pitch = PID(kp=0.046, setpoint=0)
        else:
            self.pid_yaw = PID(kp=0.03, setpoint=170)
            self.pid_z = PID(kp=4.0, setpoint=17.1)
            self.pid_x = PID(kp=12.0, setpoint=0)
            self.pid_y = PID(kp=2.45, setpoint=0)
            self.pid_roll = PID(kp=0.02, setpoint=0)  
            self.pid_pitch = PID(kp=0.046, setpoint=0)

        # CSV Logging
        self.csv_file = create_new_csv_file()
        self.init_csv()

        rospy.sleep(5)  # Wait for the guidance to be ready

        # Generate the trajectory
        self.Trajectory = Guidance(self.x0, self.y0)

        # Start control loop
        self.control_loop()

##################################### callback functions ########################################
    def imu_callback(self, msg):
        """Update roll, pitch, and yaw values from IMU data."""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(qx, qy, qz, qw)
        
        # Encrypt values if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_roll = public_key.encrypt(self.roll)
                self.safe_pitch = public_key.encrypt(self.pitch)
                self.safe_yaw = public_key.encrypt(self.yaw)
            except Exception as e:
                rospy.logerr(f"Error encrypting IMU data: {e}")

    def altitude_callback(self, msg):
        """Update altitude value."""
        self.altitude = msg.data
        
        # Encrypt altitude if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_altitude = public_key.encrypt(self.altitude)
            except Exception as e:
                rospy.logerr(f"Error encrypting altitude data: {e}")

    def pose_callback(self, msg):
        """Update x and y values from the Optitrack."""
        # The orientation of the global frame is different from the local frame
        self.x = -msg.pose.position.x
        self.y = msg.pose.position.z 
        self.time_stamp = msg.header.stamp
        
        # Encrypt position if using encryption
        if USE_ENCRYPTION:
            try:
                self.safe_x = public_key.encrypt(self.x)
                self.safe_y = public_key.encrypt(self.y)
            except Exception as e:
                rospy.logerr(f"Error encrypting position data: {e}")
                
        rospy.loginfo("New Pose received: x={:.3f}, y={:.3f}".format(self.x, self.y))

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
                # Write vehicle information as header comments
                file.write("# x0={}, y0={}, z0={}\n".format(self.x0, self.y0, self.z0))
                file.write("# roll0={}, pitch0={}, yaw0={}\n".format(self.roll0, self.pitch0, self.yaw0))
                file.write("# Starting time: {}\n".format(self.starting_time))
                
                # Write PID info
                file.write("# PID Gains:\n")
                file.write("# Roll:  kp={}\n".format(self.pid_roll.kp))
                file.write("# Pitch: kp={}\n".format(self.pid_pitch.kp))
                file.write("# Yaw:   kp={}\n".format(self.pid_yaw.kp))
                file.write("# Z:     kp={}\n".format(self.pid_z.kp))
                file.write("# X:     kp={}\n".format(self.pid_x.kp))
                file.write("# Y:     kp={}\n".format(self.pid_y.kp))
                
                # Write encryption info
                file.write("# Encryption Configuration:\n")
                file.write("# USE_ENCRYPTION: {}\n".format(USE_ENCRYPTION))
                #file.write("# ENCRYPTION_FREQUENCY: {} Hz\n".format(ENCRYPTION_FREQUENCY))
                file.write("# KEY_SIZE: {} bits\n".format(PUBLIC_KEY_SIZE))
                if USE_ENCRYPTION:
                    file.write("# ENCRYPTION_FREQUENCY: {} Hz\n".format(ENCRYPTION_FREQUENCY))
                    file.write("# KEY_SIZE: {} bits\n".format(PUBLIC_KEY_SIZE))
                    file.write("# ENCRYPT_CRITICAL_ONLY: {}\n".format(ENCRYPT_CRITICAL_ONLY))
                
                # Write header row for the CSV data - Added ComputationTime
                writer = csv.writer(file)
                writer.writerow(["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", 
                                 "Pitch", "Roll", "Yaw", "Altitude",
                                 "x", "y", "set_x", "set_y", "set_altitude", "set_roll", "set_pitch", "set_yaw",
                                 "Thruster_1", "Thruster_2", "Thruster_3", 
                                 "Thruster_4", "Thruster_5", "Thruster_6", 
                                 "Encrypted", "ComputationTime"])

    def log_to_csv(self, Fx, Fy, Fz, Mx, My, Mz, pitch, roll, yaw, altitude, x, y, set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw, thrust_forces, is_encrypted=False, computation_time=0):
        """Log data to CSV with computation time."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rospy.get_time(), Fx, Fy, Fz, Mx, My, Mz, 
                           pitch, roll, yaw, altitude, x, y, 
                           set_x, set_y, set_altitude, set_roll, set_pitch, set_yaw] 
                           + list(thrust_forces) + [1 if is_encrypted else 0, computation_time])

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
            rospy.loginfo(f"Using encryption with {PUBLIC_KEY_SIZE}-bit key at {ENCRYPTION_FREQUENCY}Hz")
            if ENCRYPT_CRITICAL_ONLY:
                rospy.loginfo("Encrypting critical attitude data only")
        
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
                
                # Start timing the PID computation
                computation_start_time = time.time()
                
                if use_encryption_this_cycle:
                    # Reset timer for next encryption cycle
                    last_encryption_time = current_time
                    rospy.loginfo("Running encrypted control cycle")
                    
                    try:
                        # Only encrypt what we need
                        if ENCRYPT_CRITICAL_ONLY:
                            # Only encrypt critical attitude data (roll/pitch)
                            safe_roll = public_key.encrypt(self.roll)
                            safe_pitch = public_key.encrypt(self.pitch)
                            safe_set_roll = public_key.encrypt(self.set_roll)
                            safe_set_pitch = public_key.encrypt(self.set_pitch)
                            
                            # Compute encrypted attitude control
                            Mx_safe = self.pid_roll.update(safe_roll, safe_set_roll)
                            My_safe = self.pid_pitch.update(safe_pitch, safe_set_pitch)
                            
                            # Non-encrypted calculations for others
                            Mz = self.pid_yaw.update(self.yaw, self.set_yaw)
                            Fz = self.pid_z.update(self.altitude, self.set_altitude)
                            Fx = self.pid_x.update(self.x, self.set_x)
                            Fy = self.pid_y.update(self.y, self.set_y)
                            
                            # Decrypt only what was encrypted
                            Mx = private_key.decrypt(Mx_safe)
                            My = private_key.decrypt(My_safe)
                        else:
                            # Encrypt all setpoints and measurements
                            safe_set_x = public_key.encrypt(self.set_x)
                            safe_set_y = public_key.encrypt(self.set_y)
                            safe_set_altitude = public_key.encrypt(self.set_altitude)
                            safe_set_roll = public_key.encrypt(self.set_roll)
                            safe_set_pitch = public_key.encrypt(self.set_pitch)
                            safe_set_yaw = public_key.encrypt(self.set_yaw)
                            
                            safe_x = public_key.encrypt(self.x)
                            safe_y = public_key.encrypt(self.y)
                            safe_roll = public_key.encrypt(self.roll)
                            safe_pitch = public_key.encrypt(self.pitch)
                            safe_yaw = public_key.encrypt(self.yaw)
                            safe_altitude = public_key.encrypt(self.altitude)
                            
                            # Compute all encrypted PID corrections
                            Mx_safe = self.pid_roll.update(safe_roll, safe_set_roll)
                            My_safe = self.pid_pitch.update(safe_pitch, safe_set_pitch)
                            Mz_safe = self.pid_yaw.update(safe_yaw, safe_set_yaw)
                            Fz_safe = self.pid_z.update(safe_altitude, safe_set_altitude)
                            Fx_safe = self.pid_x.update(safe_x, safe_set_x)
                            Fy_safe = self.pid_y.update(safe_y, safe_set_y)
                            
                            # Decrypt all PID outputs
                            Mx = private_key.decrypt(Mx_safe)
                            My = private_key.decrypt(My_safe)
                            Mz = private_key.decrypt(Mz_safe)
                            Fz = private_key.decrypt(Fz_safe)
                            Fx = private_key.decrypt(Fx_safe)
                            Fy = private_key.decrypt(Fy_safe)
                        
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
                        
                        # Also update cache
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

                # Calculate computation time in milliseconds
                computation_time = (time.time() - computation_start_time) * 1000  # Convert to milliseconds
                rospy.loginfo(f"PID computation time: {computation_time:.2f} ms")

                # Log current sensor data
                rospy.loginfo("IMU Data: Roll={:.2f}°, Pitch={:.2f}°, Yaw={:.2f}°".format(self.roll, self.pitch, self.yaw))
                rospy.loginfo("altitude: {:.2f} m".format(self.altitude))
                rospy.loginfo("Pose: x={:.3f}, y={:.3f}".format(self.x, self.y))
                rospy.loginfo("PID Outputs: Fx={:.3f}, Fy={:.3f}, Fz={:.3f}, Mx={:.3f}, My={:.3f}, Mz={:.3f}".format(Fx, Fy, Fz, Mx, My, Mz))

                # Create control vector - note that we're only applying moment control (not force control)
                tau = np.array([Fx*0, Fy*0, Fz*0, Mx, My, Mz])
                rospy.loginfo(f"Computed Control: Fx={Fx:.3f}, Fy={Fy:.3f}, Fz={Fz:.3f}, Mx={Mx:.3f}, My={My:.3f}, Mz={Mz:.3f}")

                # Compute thrust forces
                thrust_forces = np.linalg.pinv(self.tcm_enu) @ tau
                thrust_forces = np.clip(thrust_forces, -10, 15) ## -10 15 for safety reasons (put-20, 25) in the pool 
                rospy.loginfo(f"Thruster Forces: {np.round(thrust_forces, 4)}")

                # Send thruster commands
                self.thruster_manager.send_thruster_commands(thrust_forces, Delta_T=0.01)

                # Log data to CSV - include encryption status and computation time
                self.log_to_csv(Fx, Fy, Fz, Mx, My, Mz, self.pitch, self.roll, self.yaw, self.altitude, 
                               self.x, self.y, self.set_x, self.set_y, self.set_altitude, 
                               self.set_roll, self.set_pitch, self.set_yaw, thrust_forces,
                               is_encrypted=use_encryption_this_cycle,
                               computation_time=computation_time)
                               
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
                
            rate.sleep()


if __name__ == "__main__":
    try:
        ROVPIDController()
    except rospy.ROSInterruptException:
        pass