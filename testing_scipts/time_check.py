import rospy 
import numpy as np

def time_start():
    start_time = rospy.Time.now().to_sec()
    print(f"Start time: {start_time}")
    rospy.sleep(1)  # Wait for 1 second

def time_end():
    end_time = rospy.Time.now().to_sec()
    print(f"End time: {end_time}")
    rospy.sleep(1)  # Wait for 1 second

def main():
    rospy.init_node("time_check")
    time_start()
    time_end()
    rospy.spin()    

if __name__ == "__main__":
    main()
# Expected output:
# Start time: 1632951585.0
# End time: 1632951586.0
#
# Note: The actual output will vary every time you run this code.