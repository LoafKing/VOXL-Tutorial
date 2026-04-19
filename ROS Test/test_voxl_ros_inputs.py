#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

bridge = CvBridge()
got_img = False
got_pose = False

def disparity_cb(msg):
    global got_img
    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        arr = np.array(img)

        if not got_img:
            print("=== disparity topic info ===")
            print("encoding:", msg.encoding)
            print("height:", msg.height, "width:", msg.width)
            print("shape:", arr.shape)
            print("dtype:", arr.dtype)
            print("min:", np.min(arr))
            print("max:", np.max(arr))
            print("============================")
            got_img = True
    except Exception as e:
        print("disparity convert failed:", e)

def pose_cb(msg):
    global got_pose
    if not got_pose:
        p = msg.pose.position
        q = msg.pose.orientation
        print("=== qvio pose sample ===")
        print("position:", p.x, p.y, p.z)
        print("orientation:", q.x, q.y, q.z, q.w)
        print("========================")
        got_pose = True

def main():
    rospy.init_node("test_voxl_ros_inputs")
    rospy.Subscriber("/dfs_disparity", Image, disparity_cb, queue_size=1)
    rospy.Subscriber("/qvio/pose", PoseStamped, pose_cb, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()