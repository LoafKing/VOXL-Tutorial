#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class OffboardTakeoffHoverLand:
    def __init__(self):
        rospy.init_node("offboard_takeoff_hover_land_py_fixed", anonymous=True)

        # ====== Parameters ======
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)

        # 改成“相对当前高度上升多少米”
        self.takeoff_height_rel = rospy.get_param("~takeoff_height_rel", 0.8)

        self.hover_time = rospy.get_param("~hover_time", 8.0)
        self.preflight_setpoint_count = rospy.get_param("~preflight_setpoint_count", 100)
        self.land_xy_hold = rospy.get_param("~land_xy_hold", True)
        self.position_reached_tol = rospy.get_param("~position_reached_tol", 0.12)
        self.land_disarm_delay = rospy.get_param("~land_disarm_delay", 2.0)
        self.startup_delay = rospy.get_param("~startup_delay", 5.0)

        self.offboard_wait_timeout = rospy.get_param("~offboard_wait_timeout", 5.0)
        self.arm_wait_timeout = rospy.get_param("~arm_wait_timeout", 8.0)
        self.takeoff_timeout = rospy.get_param("~takeoff_timeout", 15.0)
        self.land_timeout = rospy.get_param("~land_timeout", 8.0)

        self.ground_z_margin = rospy.get_param("~ground_z_margin", 0.05)
        self.verbose_climb_log_interval = rospy.get_param("~verbose_climb_log_interval", 1.0)

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

    def set_offboard_mode_once(self):
        rospy.loginfo("Requesting OFFBOARD mode...")
        try:
            resp = self.set_mode_client(0, "OFFBOARD")
            rospy.loginfo(f"OFFBOARD mode_sent: {resp.mode_sent}")
            return resp.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to set OFFBOARD: {e}")
            return False

    def arm_once(self, value=True):
        rospy.loginfo(f"{'Arming' if value else 'Disarming'} request...")
        try:
            resp = self.arming_client(value)
            rospy.loginfo(f"Arm success: {resp.success}, result: {resp.result}")
            return resp.success, resp.result
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service failed: {e}")
            return False, -1

    def dist_to_target(self, target: PoseStamped):
        if self.current_pose is None:
            return float("inf")
        dx = self.current_pose.pose.position.x - target.pose.position.x
        dy = self.current_pose.pose.position.y - target.pose.position.y
        dz = self.current_pose.pose.position.z - target.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def current_z(self):
        if self.current_pose is None:
            return float("nan")
        return float(self.current_pose.pose.position.z)

    def confirm_mode(self, desired_mode: str, target: PoseStamped, timeout_sec: float):
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish_target(target)

            if self.current_state.mode == desired_mode:
                rospy.loginfo(f"Confirmed mode: {desired_mode}")
                return True

            if (rospy.Time.now() - start).to_sec() > timeout_sec:
                rospy.logwarn(
                    f"Timeout waiting for mode={desired_mode}. Current mode={self.current_state.mode}"
                )
                return False

            self.rate.sleep()
        return False

    def confirm_armed(self, desired_armed: bool, target: PoseStamped, timeout_sec: float):
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            self.publish_target(target)

            if self.current_state.armed == desired_armed:
                rospy.loginfo(f"Confirmed armed={desired_armed}")
                return True

            if (rospy.Time.now() - start).to_sec() > timeout_sec:
                rospy.logwarn(
                    f"Timeout waiting for armed={desired_armed}. Current armed={self.current_state.armed}"
                )
                return False

            self.rate.sleep()
        return False

    def set_offboard_and_confirm(self, target: PoseStamped):
        self.set_offboard_mode_once()
        return self.confirm_mode("OFFBOARD", target, self.offboard_wait_timeout)

    def arm_and_confirm(self, target: PoseStamped):
        self.arm_once(True)
        return self.confirm_armed(True, target, self.arm_wait_timeout)

    def disarm_and_confirm(self, target: PoseStamped):
        self.arm_once(False)
        return self.confirm_armed(False, target, 5.0)

    def hold_until_reached_or_timeout(self, target: PoseStamped, timeout_sec: float):
        start = rospy.Time.now()
        last_log_t = 0.0

        while not rospy.is_shutdown():
            self.publish_target(target)

            dist = self.dist_to_target(target)
            now_t = (rospy.Time.now() - start).to_sec()

            if dist < self.position_reached_tol:
                rospy.loginfo(f"Reached target within tolerance: {dist:.3f} m")
                return True

            if now_t - last_log_t >= self.verbose_climb_log_interval:
                last_log_t = now_t
                rospy.loginfo(
                    f"[climb] mode={self.current_state.mode}, armed={self.current_state.armed}, "
                    f"current_z={self.current_z():.3f}, target_z={target.pose.position.z:.3f}, dist={dist:.3f}"
                )

            if now_t > timeout_sec:
                rospy.logwarn(
                    f"Timeout before fully reaching target. "
                    f"mode={self.current_state.mode}, armed={self.current_state.armed}, "
                    f"current_z={self.current_z():.3f}, target_z={target.pose.position.z:.3f}, dist={dist:.3f}"
                )
                return False

            self.rate.sleep()

        return False

    def hold_for_duration(self, target: PoseStamped, duration_sec: float, label="Holding"):
        rospy.loginfo(f"{label} for {duration_sec:.1f} s")
        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < duration_sec:
            self.publish_target(target)
            self.rate.sleep()

    def abort_mission(self, safe_target: PoseStamped, reason: str):
        rospy.logerr(f"Mission aborted: {reason}")

        # 持续发一个安全目标几秒，避免立刻断 setpoint
        end_t = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - end_t).to_sec() < 2.0:
            self.publish_target(safe_target)
            self.rate.sleep()

    def main(self):
        self.wait_for_connection_and_pose()

        # Use current XY and current pose as reference
        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y
        start_z = self.current_pose.pose.position.z
        start_orientation = self.current_pose.pose.orientation

        rospy.loginfo(
            f"Current pose: x={start_x:.3f}, y={start_y:.3f}, z={start_z:.3f}"
        )

        if self.startup_delay > 0.0:
            rospy.loginfo(f"Startup delay: waiting {self.startup_delay:.1f} s before mission start...")
            rospy.sleep(self.startup_delay)

        # ===== Phase 1: Takeoff target =====
        target_z = start_z + self.takeoff_height_rel
        takeoff_target = self.make_target(
            x=start_x,
            y=start_y,
            z=target_z,
            orientation=start_orientation
        )

        rospy.loginfo(
            f"Takeoff target set to relative climb: current_z={start_z:.3f} -> target_z={target_z:.3f}"
        )

        self.pre_send_setpoints(takeoff_target)

        # Switch to OFFBOARD and confirm
        if self.current_state.mode != "OFFBOARD":
            ok_offboard = self.set_offboard_and_confirm(takeoff_target)
        else:
            rospy.loginfo("Already in OFFBOARD mode.")
            ok_offboard = True

        if not ok_offboard:
            self.abort_mission(takeoff_target, "Failed to enter OFFBOARD")
            return

        # Keep publishing a bit before arming
        self.hold_for_duration(takeoff_target, 1.5, label="Stabilizing setpoints before arming")

        # Arm and confirm
        if not self.current_state.armed:
            ok_arm = self.arm_and_confirm(takeoff_target)
        else:
            rospy.loginfo("Vehicle already armed.")
            ok_arm = True

        if not ok_arm:
            self.abort_mission(takeoff_target, "Failed to arm")
            return

        rospy.loginfo("Vehicle is confirmed ARMED and in OFFBOARD.")

        # ===== Phase 2: Takeoff / climb =====
        reached = self.hold_until_reached_or_timeout(takeoff_target, timeout_sec=self.takeoff_timeout)
        if not reached:
            rospy.logwarn("Takeoff target was not fully reached, but continuing to hover/land sequence safely.")

        # ===== Phase 3: Hover =====
        self.hold_for_duration(takeoff_target, self.hover_time, label="Hovering")

        # ===== Phase 4: Land =====
        if self.land_xy_hold:
            land_x = start_x
            land_y = start_y
            land_orientation = start_orientation
        else:
            land_x = self.current_pose.pose.position.x
            land_y = self.current_pose.pose.position.y
            land_orientation = self.current_pose.pose.orientation

        land_target = self.make_target(
            x=land_x,
            y=land_y,
            z=max(self.ground_z_margin, start_z),
            orientation=land_orientation
        )

        rospy.loginfo(
            f"Landing by commanding local z back near ground: target_z={land_target.pose.position.z:.3f}"
        )
        self.hold_for_duration(land_target, self.land_timeout, label="Descending")
        self.hold_for_duration(land_target, self.land_disarm_delay, label="Ground hold before disarm")

        # ===== Phase 5: Disarm =====
        ok_disarm = self.disarm_and_confirm(land_target)
        if not ok_disarm:
            rospy.logwarn("Disarm command did not get confirmed.")
        else:
            rospy.loginfo("Vehicle disarmed successfully.")

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