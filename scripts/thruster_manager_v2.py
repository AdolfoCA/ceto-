from mavros_msgs.msg import OverrideRCIn, State, RCOut
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from pwm_fitfit_function import load_data, fit_polynomials, force_to_pwm
from direct_rc import blueROV1_matrix
import pandas as pd
import numpy as np
import rospy
import os

class ThrusterManager:
    def __init__(self):
        
        if not rospy.core.is_initialized():
            rospy.init_node("thruster_manager", anonymous=True)

        self.pub = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=2)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.rc_out_sub = rospy.Subscriber("/mavros/rc/out", RCOut, self.rc_out_callback)  

        self.rate = rospy.Rate(100)  # 100 Hz
        self.current_state = None

        # Load and process the data 
        #cwd = os.path.dirname(os.path.abspath(__file__))
        #file_path = os.path.join(cwd, "..", "data", "T200_data_16V.csv")
        file_path = "T20016V.csv"
        self.data = load_data(file_path)

        # Fit a 4th-degree polynomial 
        self.coeffs_negative, self.coeffs_positive = fit_polynomials(self.data, 4)

    def state_callback(self, msg):
        """Callback to check the state."""
        self.current_state = msg

    def rc_out_callback(self, msg):
        """Callback to store the latest /mavros/rc/out message."""
        self.last_rc_out = msg.channels  

    def get_rc_out(self):
        """Return the latest /mavros/rc/out values."""
        return self.last_rc_out if self.last_rc_out is not None else [0] * 18  

    def apply_transformation(self, pwm_values):
        """Applies the transformation from direct_rc.py."""
        adjusted_values = pwm_values - 1500
        blueROV1_transposed_inverse = np.linalg.inv(blueROV1_matrix.T)
        transformed_values = np.dot(adjusted_values, blueROV1_transposed_inverse)
        return transformed_values + 1500

    def arm_vehicle(self):
        rospy.wait_for_service("/mavros/cmd/arming")
        try:
            arm_service = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
            req = CommandBoolRequest()
            req.value = True
            resp = arm_service(req)
            if resp.success:
                rospy.loginfo("Vehicle armed successfully.")
            else:
                rospy.logwarn("Failed to arm vehicle.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service call failed: {e}")
    
    def disarm_vehicle(self):
        rospy.wait_for_service("/mavros/cmd/arming")
        try:
            arm_service = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
            req = CommandBoolRequest()
            req.value = False  # False = disarm
            resp = arm_service(req)
            if resp.success:
                rospy.loginfo("Vehicle disarmed successfully.")
                print("")
            else:
                rospy.logwarn("Failed to disarm vehicle.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Disarming service call failed: {e}")

    def set_manual_mode(self):
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


    def send_thruster_commands(self, N, Delta_T):
        """Convert forces to PWM, apply transformation, and send commands for Delta_T."""

        #rospy.loginfo("Waiting for MAVROS state update...")
        while self.current_state is None and not rospy.is_shutdown():
            rospy.sleep(0.01)

        if not self.current_state.armed:
            rospy.loginfo("Arming vehicle...")
            self.arm_vehicle()

        if self.current_state.mode != "MANUAL":
            rospy.loginfo("Setting mode to MANUAL...")
            self.set_manual_mode()

        # Convert forces to PWM using precomputed coefficients and handle dead zone
        pwm_values = np.array([
            force_to_pwm(self.coeffs_negative, self.coeffs_positive, N[i])
            for i in range(6)
        ])
        # Ensure PWM is within safe range (1100 - 1900), otherwise set to 1500
        pwm_values = np.where((pwm_values < 1100) | (pwm_values > 1900), 1500, pwm_values)
        rospy.loginfo(f"Converted PWM values: {pwm_values}")

        # Apply transformation
        pwm_transformed = self.apply_transformation(pwm_values)
        # rospy.loginfo(f"Transformed PWM values: {np.round(pwm_transformed, 2).tolist()}")

        # Prepare and publish the message
        msg = OverrideRCIn()
        msg.channels[:6] = pwm_transformed.astype(int).tolist()
        msg.channels[6:] = [0] * (18 - 6)  

        start_time = rospy.Time.now()
        rate = rospy.Rate(100)
        while (rospy.Time.now() - start_time).to_sec() < Delta_T and not rospy.is_shutdown():
            self.pub.publish(msg)
            self.rate.sleep()
            #rospy.loginfo(f"RC Out Values: {self.get_rc_out()}")

        # Return to neutral PWM
        #msg.channels[:6] = [1500] * 6
        #rospy.loginfo("Returning to neutral PWM...")
        #self.pub.publish(msg)
        #start_time = rospy.Time.now()
        #while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
        #self.pub.publish(msg)
        #self.rate.sleep()
        #