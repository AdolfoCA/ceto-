#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def fake_pose_publisher():
    # Initialize the ROS node
    rospy.init_node('fake_pose_publisher', anonymous=True)
    
    # Create a publisher for the PoseStamped message
    pose_pub = rospy.Publisher('/vrpn_client_node/Galene/pose', PoseStamped, queue_size=10)
    
    # Set the publishing rate to 10 Hz
    rate = rospy.Rate(10)  # 10 Hz
    
    # Message to publish
    pose_msg = PoseStamped()
    
    # Set the frame ID
    pose_msg.header.frame_id = "world"
    
    # Set the position (x=5, y=5, z=0)
    pose_msg.pose.position.x = 5.0
    pose_msg.pose.position.y = 5.0
    pose_msg.pose.position.z = 0.0
    
    # Set the orientation (identity quaternion)
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0
    
    rospy.loginfo("Starting fake pose publisher at 10Hz with position (5, 5, 0)")
    
    # Main loop
    while not rospy.is_shutdown():
        # Update timestamp to current time
        pose_msg.header.stamp = rospy.Time.now()
        
        # Publish the message
        pose_pub.publish(pose_msg)
        
        # Sleep to maintain the desired rate
        rate.sleep()

if __name__ == '__main__':
    try:
        fake_pose_publisher()
    except rospy.ROSInterruptException:
        pass