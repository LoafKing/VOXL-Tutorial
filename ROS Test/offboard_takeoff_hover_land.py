#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class OffboardTakeoffHoverLand:
    def __init__(self):
        rospy.init_node("offboard_takeoff_hover_land_py", anonymous=True)

        # ====== Parameters ======
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.0)   # meters in local frame
        self.hover_time = rospy.get_param("~hover_time", 8.0)           # seconds
        self.preflight_setpoint_count = rospy.get_param("~preflight_setpoint_count", 100)
        self.land_xy_hold = rospy.get_param("~land_xy_hold", True)
        self.position_reached_tol = rospy.get_param("~position_reached_tol", 0.12)
        self.land_disarm_delay = rospy.get_param("~land_disarm_delay", 2.0)

        # ====== State ======
        self.current_state = State()
        self.current_pose = None
        self.connected = False

        # ====== ROS interfaces ======
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=10)
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

        rospy.loginfo("Waiting for MAVROS services...")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.rate = rospy.Rate(self.rate_hz)

    def state_cb(self, msg: State):
        self.current_state = msg
        self.connected = msg.connected

    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def make_target(self, x, y, z, orientation=None):
        target = PoseStamped()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"

        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.position.z = float(z)

        if orientation is not None:
            target.pose.orientation = orientation
        else:
            target.pose.orientation.w = 1.0

        return target

    def publish_target(self, target: PoseStamped):
        target.header.stamp = rospy.Time.now()
        self.local_pos_pub.publish(target)

    def wait_for_connection_and_pose(self):
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.connected:
            self.rate.sleep()

        rospy.loginfo("FCU connected. Waiting for local position...")
        while not rospy.is_shutdown() and self.current_pose is None:
            self.rate.sleep()

        rospy.loginfo("Local position received.")

    def pre_send_setpoints(self, target: PoseStamped):
        rospy.loginfo(f"Pre-sending setpoints ({self.preflight_setpoint_count} msgs)...")
        for _ in range(int(self.preflight_setpoint_count)):
            if rospy.is_shutdown():
                return
            self.publish_target(target)
            self.rate.sleep()

    def set_offboard_mode(self):
        rospy.loginfo("Setting OFFBOARD mode...")
        try:
            resp = self.set_mode_client(0, "OFFBOARD")
            rospy.loginfo(f"OFFBOARD mode_sent: {resp.mode_sent}")
            return resp.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to set OFFBOARD: {e}")
            return False

    def arm(self, value=True):
        rospy.loginfo(f"{'Arming' if value else 'Disarming'}...")
        try:
            resp = self.arming_client(value)
            rospy.loginfo(f"Arm success: {resp.success}, result: {resp.result}")
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service failed: {e}")
            return False

    def dist_to_target(self, target: PoseStamped):
        if self.current_pose is None:
            return float("inf")
        dx = self.current_pose.pose.position.x - target.pose.position.x
        dy = self.current_pose.pose.position.y - target.pose.position.y
        dz = self.current_pose.pose.position.z - target.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def hold_until_reached_or_timeout(self, target: PoseStamped, timeout_sec: float):
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish_target(target)

            dist = self.dist_to_target(target)
            if dist < self.position_reached_tol:
                rospy.loginfo(f"Reached target within tolerance: {dist:.3f} m")
                return True

            if (rospy.Time.now() - start).to_sec() > timeout_sec:
                rospy.logwarn(f"Timeout before fully reaching target. Current dist={dist:.3f} m")
                return False

            self.rate.sleep()

    def hold_for_duration(self, target: PoseStamped, duration_sec: float, label="Holding"):
        rospy.loginfo(f"{label} for {duration_sec:.1f} s")
        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < duration_sec:
            self.publish_target(target)
            self.rate.sleep()

    def main(self):
        self.wait_for_connection_and_pose()

        # Use current XY as takeoff reference
        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y
        start_z = self.current_pose.pose.position.z
        start_orientation = self.current_pose.pose.orientation

        rospy.loginfo(
            f"Current pose: x={start_x:.3f}, y={start_y:.3f}, z={start_z:.3f}"
        )

        # ===== Phase 1: Takeoff target =====
        takeoff_target = self.make_target(
            x=start_x,
            y=start_y,
            z=self.takeoff_height,
            orientation=start_orientation
        )

        self.pre_send_setpoints(takeoff_target)

        # Switch to OFFBOARD
        if self.current_state.mode != "OFFBOARD":
            self.set_offboard_mode()

        # Keep publishing a bit before arming
        self.hold_for_duration(takeoff_target, 1.5, label="Stabilizing setpoints before arming")

        # Arm
        if not self.current_state.armed:
            self.arm(True)

        # Wait until state reflects armed + offboard, while still publishing
        wait_start = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish_target(takeoff_target)

            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                rospy.loginfo("Vehicle is armed and in OFFBOARD.")
                break

            if (rospy.Time.now() - wait_start).to_sec() > 8.0:
                rospy.logwarn(
                    f"State not fully reached yet. mode={self.current_state.mode}, armed={self.current_state.armed}"
                )
                break

            self.rate.sleep()

        # ===== Phase 2: Takeoff / climb =====
        self.hold_until_reached_or_timeout(takeoff_target, timeout_sec=15.0)

        # ===== Phase 3: Hover =====
        self.hold_for_duration(takeoff_target, self.hover_time, label="Hovering")

        # ===== Phase 4: Land =====
        if self.land_xy_hold:
            land_target = self.make_target(
                x=start_x,
                y=start_y,
                z=max(0.05, start_z),
                orientation=start_orientation
            )
        else:
            # Hold current XY at the moment landing starts
            land_target = self.make_target(
                x=self.current_pose.pose.position.x,
                y=self.current_pose.pose.position.y,
                z=max(0.05, start_z),
                orientation=self.current_pose.pose.orientation
            )

        rospy.loginfo("Landing by commanding local z back near ground...")
        self.hold_for_duration(land_target, 8.0, label="Descending")

        # Optional extra hold near ground
        self.hold_for_duration(land_target, self.land_disarm_delay, label="Ground hold before disarm")

        # ===== Phase 5: Disarm =====
        self.arm(False)

        rospy.loginfo("Mission complete.")

        # Publish a little longer so mode transition is smooth
        end_start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - end_start).to_sec() < 1.0:
            self.publish_target(land_target)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = OffboardTakeoffHoverLand()
        node.main()
    except rospy.ROSInterruptException:
        pass