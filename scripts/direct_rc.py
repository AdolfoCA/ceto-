import numpy as np
import rospy
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

# Define the blueROV1_matrix
blueROV1_matrix = np.array([
    [0.0,  0.0,  0.0,  1.0,-1.0, 0.0], # Motor 1 
    [0.0,  0.0,  0.0,  1.0, 1.0, 0.0], # Motor 2 
    [0.5, -0.5,  0.45, 0.0, 0.0, 0.0], # Motor 3
    [0.5,  0.5,  0.45, 0.0, 0.0, 0.0], # Motor 4 
    [-1.0, 0.0,  1.0,  0.0, 0.0, 0.0], # Motor 5 
    [0.0,  0.25, 0.0,  0.0, 0.0,-1.0]  # Motor 6
])

def arm_vehicle():
    rospy.wait_for_service("/mavros/cmd/arming")
    try:
        arm_service = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        req = CommandBoolRequest()
        req.value = True  # True = arm
        resp = arm_service(req)
        if resp.success:
            rospy.loginfo("Vehicle armed successfully.")
        else:
            rospy.logwarn("Failed to arm vehicle.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Arming service call failed: {e}")

def disarm_vehicle():
    rospy.wait_for_service("/mavros/cmd/arming")
    try:
        arm_service = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        req = CommandBoolRequest()
        req.value = False  # False = disarm
        resp = arm_service(req)
        if resp.success:
            rospy.loginfo("Vehicle disarmed successfully.")
        else:
            rospy.logwarn("Failed to disarm vehicle.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Disarming service call failed: {e}")

def set_manual_mode():
    rospy.wait_for_service('/mavros/set_mode')
    try:
        set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        req = SetModeRequest()
        req.custom_mode = "MANUAL"  
        resp = set_mode_service(req)
        if resp.mode_sent:
            rospy.loginfo("Flight mode changed to MANUAL successfully.")
        else:
            rospy.logwarn("Failed to change flight mode to MANUAL.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Setting flight mode failed: {e}")

def override_rc():
    rospy.init_node("rc_ovrd")
    pub = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=10)
    rate = rospy.Rate(100)
    
    msg0 = OverrideRCIn()
    msg = OverrideRCIn()

    # Default all channels to neutral (1500)
    msg0.channels = [1500, 1500, 1500, 1500, 1500, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    msg.channels = [1500, 1500, 1500, 1500, 1500, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    rospy.loginfo("Arming vehicle...")
    arm_vehicle()

    rospy.loginfo("Setting flight mode to MANUAL...")
    set_manual_mode()
    
    rospy.loginfo("Publishing RC override...")

    # Take the first six values of msg.channels
    original_values = np.array(msg.channels[:6])
    rospy.loginfo(f"Original values: {original_values}")

    # Subtract 1500
    adjusted_values = original_values - 1500

    # Compute the inverse of the transpose of blueROV1_matrix
    blueROV1_transposed_inverse = np.linalg.inv(blueROV1_matrix.T)

    # Multiply adjusted values by the inverse of the transpose
    transformed_values = np.dot(adjusted_values, blueROV1_transposed_inverse)

    # Add 1500 to the result
    final_values = transformed_values + 1500

    # Replace the first six values in msg.channels with the transformed values
    msg.channels[:6] = final_values.astype(int)
    rospy.loginfo(f"Transformed values: {msg.channels[:6]}")

    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time).to_sec() < 2.0:
        pub.publish(msg)
        rate.sleep()
    
    rospy.loginfo("Returning to neutral PWM...")
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time).to_sec() < 2.0:
        pub.publish(msg0)
        rate.sleep()

    rospy.loginfo("Disarming vehicle for safety...")
    disarm_vehicle()

    rospy.loginfo("RC override complete.")

if __name__ == "__main__":
    try:
        override_rc()
    except rospy.ROSInterruptException:
        pass
