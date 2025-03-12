#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_matrix, euler_from_quaternion
import math

class IMUProcessor:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('imu_processor', anonymous=True)
        
        # Subscribe to IMU data
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        
        # Variables to store processed data
        self.rotation_matrix = np.identity(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        rospy.loginfo("IMU Processor initialized. Waiting for IMU data...")
        
    def imu_callback(self, msg):
        # Extract quaternion from the IMU message
        quat = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        
        # Calculate rotation matrix from quaternion
        # quaternion_matrix returns a 4x4 homogeneous transformation matrix
        # We need to extract the 3x3 rotation part
        homogeneous_matrix = quaternion_matrix(quat)
        self.rotation_matrix = homogeneous_matrix[:3, :3]
        
        # Calculate roll, pitch, yaw angles from quaternion
        # euler_from_quaternion returns (roll, pitch, yaw) in radians
        euler_angles = euler_from_quaternion(quat)
        self.roll = euler_angles[0]
        self.pitch = euler_angles[1]
        self.yaw = euler_angles[2]
        
        # Convert angles to degrees for display
        roll_deg = math.degrees(self.roll)
        pitch_deg = math.degrees(self.pitch)
        yaw_deg = math.degrees(self.yaw)
        
        # Print the results
        rospy.loginfo("Rotation Matrix:\n%s", self.rotation_matrix)
        rospy.loginfo("Angles (radians) - Roll: %.4f, Pitch: %.4f, Yaw: %.4f", 
                      self.roll, self.pitch, self.yaw)
        rospy.loginfo("Angles (degrees) - Roll: %.4f, Pitch: %.4f, Yaw: %.4f", 
                      roll_deg, pitch_deg, yaw_deg)
        
    def run(self):
        # Keep the node running
        rospy.spin()

# Alternative quaternion to rotation matrix function for reference
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix manually.
    
    Args:
        q: A quaternion [x, y, z, w]
    
    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q
    
    # Check if quaternion is normalized
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1.0) > 1e-6:
        # Normalize the quaternion
        w /= norm
        x /= norm
        y /= norm
        z /= norm
    
    # Compute rotation matrix elements
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

if __name__ == '__main__':
    try:
        imu_processor = IMUProcessor()
        imu_processor.run()
    except rospy.ROSInterruptException:
        pass