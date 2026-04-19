import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import rospy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from mavros_msgs.msg import State
    from geometry_msgs.msg import Twist
except Exception:
    rospy = None
    PoseStamped = object
    TwistStamped = object
    Imu = object
    Odometry = object
    State = object
    Twist = object


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@dataclass
class RealEnvConfig:
    rtsp_url: str = "rtsp://127.0.0.1:8900/live"
    crop_size: int = 96
    frame_stack_k: int = 4
    use_recent_visit_map: bool = True
    use_proxy_context: bool = True

    # Action scaling: keep consistent with MuJoCo env
    vxy_max: float = 1.0
    yaw_rate_max_deg: float = 120.0
    z_target: float = 0.5
    safe_margin_m: float = 0.40

    # World/grid settings for local top-down patch
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    res_xy: float = 0.10

    # Recent-visit memory
    recent_visit_tau_steps: float = 80.0
    recent_visit_binary: bool = False
    recent_visit_binary_steps: int = 80

    # Camera/depth quality proxies
    proxy_ema_alpha: float = 0.15
    depth_valid_ratio_thr: float = 0.01

    # Looping
    control_rate_hz: float = 10.0
    command_topic: str = "/mavros/setpoint_velocity/cmd_vel_unstamped"
    pose_topic: str = "/mavros/local_position/pose"
    vel_topic: str = "/mavros/local_position/velocity_local"
    imu_topic: str = "/mavros/imu/data"
    odom_topic: str = "/mavros/local_position/odom"
    state_topic: str = "/mavros/state"


class LatestFrameReader:
    """Background RTSP reader that always keeps the newest frame only."""

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        assert self.cap is not None
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.latest_bgr = frame
                self.latest_ts = time.time()

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_bgr is None:
                return None, 0.0
            return self.latest_bgr.copy(), float(self.latest_ts)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RealDroneEnvMinimal(gym.Env):
    """
    Minimal real-world single-agent environment matching the newest MuJoCo env's
    observation/action semantics as closely as possible.

    This is intentionally a deployment/inference env, not a training env.
    Main design goals:
      1. Keep action meaning identical to MuJoCo env: normalized [vx, vy, yaw_rate].
      2. Keep observation structure identical in shape and ordering:
         image: [belief, known, obstacle, neighbor(zeros), recent_visit] * frame_stack_k
         state: [vx, vy, vz, sin(yaw), cos(yaw), wz, z_err, d, gx, gy, qvis, qvio]
      3. Replace sim-only internals with real sensor backends.

    You must still plug in your real depth/ESDF/map pipeline where marked TODO.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: RealEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.num_agents = 1
        self.crop_size = int(cfg.crop_size)
        self.frame_stack_k = int(max(1, cfg.frame_stack_k))
        self.use_recent_visit_map = bool(cfg.use_recent_visit_map)
        self.use_proxy_context = bool(cfg.use_proxy_context)
        self.proxy_context_dim = 2 if self.use_proxy_context else 0

        self.yaw_rate_max = np.deg2rad(float(cfg.yaw_rate_max_deg))
        self.dt_step = 1.0 / max(1e-6, float(cfg.control_rate_hz))

        # Global grid storage for real-world map-like observation building.
        self.map_h = int(np.ceil((cfg.y_max - cfg.y_min) / cfg.res_xy))
        self.map_w = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.res_xy))
        self._global_seen_mask = np.zeros((self.map_h, self.map_w), dtype=bool)
        self._global_obstacle_mask = np.zeros((self.map_h, self.map_w), dtype=bool)
        self._belief = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self._belief[:] = 1.0 / float(max(1, self.map_h * self.map_w))

        self._visit_age_max = int(np.iinfo(np.int32).max // 4)
        self._recent_init_age = int(
            max(cfg.recent_visit_binary_steps * 4, round(cfg.recent_visit_tau_steps * 4.0), 4)
        )
        self._self_visit_age = np.full((self.map_h, self.map_w), self._recent_init_age, dtype=np.int32)

        base_img_channels = 5 if self.use_recent_visit_map else 4
        self.local_img_channels = base_img_channels * self.frame_stack_k
        state_dim = 10 + self.proxy_context_dim

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 3), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(1, self.local_img_channels, self.crop_size, self.crop_size), dtype=np.uint8
                ),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(1, state_dim), dtype=np.float32),
                "state_cbf": spaces.Box(low=-np.inf, high=np.inf, shape=(1, state_dim), dtype=np.float32),
            }
        )

        # ROS state cache.
        self._pose_msg: Optional[PoseStamped] = None
        self._vel_msg: Optional[TwistStamped] = None
        self._imu_msg: Optional[Imu] = None
        self._odom_msg: Optional[Odometry] = None
        self._mav_state: Optional[State] = None
        self._last_pose_ts: float = 0.0
        self._last_vel_ts: float = 0.0
        self._last_frame_ts: float = 0.0

        self.depth_valid_ema: float = 1.0
        self.vio_consistency_ema: float = 1.0
        self._prev_pose_for_vio_proxy: Optional[Tuple[float, float, float, float, float]] = None
        self._img_history: Deque[np.ndarray] = deque(maxlen=self.frame_stack_k)

        self.frame_reader = LatestFrameReader(cfg.rtsp_url)
        self._cmd_pub = None
        self._ros_ready = False

        self._setup_ros()
        self.frame_reader.start()

    # ---------------------------------------------------------------------
    # ROS init and callbacks
    # ---------------------------------------------------------------------
    def _setup_ros(self) -> None:
        if rospy is None:
            raise RuntimeError("rospy is unavailable. Run this on a ROS machine with mavros installed.")
        if not rospy.core.is_initialized():
            rospy.init_node("real_drone_env_minimal", anonymous=True)

        rospy.Subscriber(self.cfg.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.cfg.vel_topic, TwistStamped, self._vel_cb, queue_size=1)
        rospy.Subscriber(self.cfg.imu_topic, Imu, self._imu_cb, queue_size=1)
        rospy.Subscriber(self.cfg.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(self.cfg.state_topic, State, self._state_cb, queue_size=1)
        self._cmd_pub = rospy.Publisher(self.cfg.command_topic, Twist, queue_size=1)
        self._ros_ready = True

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._pose_msg = msg
        self._last_pose_ts = time.time()

    def _vel_cb(self, msg: TwistStamped) -> None:
        self._vel_msg = msg
        self._last_vel_ts = time.time()

    def _imu_cb(self, msg: Imu) -> None:
        self._imu_msg = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_msg = msg

    def _state_cb(self, msg: State) -> None:
        self._mav_state = msg

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _world_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        j = int(np.floor((x - self.cfg.x_min) / self.cfg.res_xy))
        i = int(np.floor((y - self.cfg.y_min) / self.cfg.res_xy))
        return i, j

    def _crop(self, arr2d: np.ndarray, ci: int, cj: int) -> np.ndarray:
        pad = self.crop_size // 2 + 2
        a = np.pad(arr2d, ((pad, pad), (pad, pad)), mode="constant")
        I, J = int(ci) + pad, int(cj) + pad
        return a[I - self.crop_size // 2:I + self.crop_size // 2, J - self.crop_size // 2:J + self.crop_size // 2]

    def _visit_age_to_recent(self, age_map: np.ndarray) -> np.ndarray:
        if self.cfg.recent_visit_binary:
            return (age_map < float(self.cfg.recent_visit_binary_steps)).astype(np.float32)
        return np.exp(-age_map / float(self.cfg.recent_visit_tau_steps)).astype(np.float32)

    def _advance_recent_visit(self, x: float, y: float) -> None:
        np.minimum(self._self_visit_age + 1, self._visit_age_max, out=self._self_visit_age)
        ci, cj = self._world_to_ij(x, y)
        if 0 <= ci < self.map_h and 0 <= cj < self.map_w:
            self._self_visit_age[ci, cj] = 0

    def _get_pose_xyzyaw(self) -> Tuple[float, float, float, float]:
        if self._pose_msg is None:
            raise RuntimeError("No pose received from MAVROS yet.")
        p = self._pose_msg.pose.position
        q = self._pose_msg.pose.orientation
        yaw = quat_xyzw_to_yaw(q.x, q.y, q.z, q.w)
        return float(p.x), float(p.y), float(p.z), float(yaw)

    def _get_velocity(self) -> Tuple[float, float, float]:
        if self._vel_msg is not None:
            v = self._vel_msg.twist.linear
            return float(v.x), float(v.y), float(v.z)
        if self._odom_msg is not None:
            v = self._odom_msg.twist.twist.linear
            return float(v.x), float(v.y), float(v.z)
        return 0.0, 0.0, 0.0

    def _get_wz(self) -> float:
        if self._imu_msg is not None:
            return float(self._imu_msg.angular_velocity.z)
        if self._odom_msg is not None:
            return float(self._odom_msg.twist.twist.angular.z)
        return 0.0

    def _compute_qvis(self, frame_bgr: Optional[np.ndarray]) -> float:
        if frame_bgr is None:
            raw = 0.0
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            valid = np.isfinite(gray)
            raw = float(np.count_nonzero(valid)) / float(max(1, gray.size))
        self.depth_valid_ema = (1.0 - self.cfg.proxy_ema_alpha) * self.depth_valid_ema + self.cfg.proxy_ema_alpha * raw
        return float(np.clip(self.depth_valid_ema, 0.0, 1.0))

    def _compute_qvio(self, x: float, y: float, z: float, yaw: float) -> float:
        now = time.time()
        if self._prev_pose_for_vio_proxy is None:
            self._prev_pose_for_vio_proxy = (x, y, z, yaw, now)
            self.vio_consistency_ema = 1.0
            return 1.0
        px, py, pz, pyaw, pt = self._prev_pose_for_vio_proxy
        dt = max(1e-6, now - pt)
        dx, dy, dz = x - px, y - py, z - pz
        dyaw = wrap_to_pi(yaw - pyaw)

        lin_jump = np.sqrt(dx * dx + dy * dy + dz * dz) / dt
        yaw_jump = abs(dyaw) / dt

        lin_score = float(np.clip(1.0 - lin_jump / 3.0, 0.0, 1.0))
        yaw_score = float(np.clip(1.0 - yaw_jump / np.deg2rad(180.0), 0.0, 1.0))
        raw = 0.5 * (lin_score + yaw_score)
        self.vio_consistency_ema = (1.0 - self.cfg.proxy_ema_alpha) * self.vio_consistency_ema + self.cfg.proxy_ema_alpha * raw
        self._prev_pose_for_vio_proxy = (x, y, z, yaw, now)
        return float(np.clip(self.vio_consistency_ema, 0.0, 1.0))

    def _query_local_safety(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Placeholder for real-world safety features: (d, gx, gy).

        Replace this with your real ESDF / depth-field query.
        For now we return a conservative default:
          d  = safe_margin
          gx = 0
          gy = 0
        """
        d = float(self.cfg.safe_margin_m)
        gx = 0.0
        gy = 0.0
        return d, gx, gy

    def _update_real_maps_from_sensors(self, frame_bgr: Optional[np.ndarray], x: float, y: float, yaw: float) -> None:
        """
        Minimal placeholder for map updates.

        You should replace this with your real known/obstacle update pipeline using:
          - depth / stereo / ESDF / occupancy
          - FOV projection into the global grid

        Current minimal behavior:
          - mark current cell as seen
          - keep obstacle map unchanged
        """
        ci, cj = self._world_to_ij(x, y)
        if 0 <= ci < self.map_h and 0 <= cj < self.map_w:
            self._global_seen_mask[ci, cj] = True

    def _build_image_obs(self, x: float, y: float) -> np.ndarray:
        ci, cj = self._world_to_ij(x, y)

        belief_mean = float(self._belief.mean())
        belief_vis_full = np.where(self._global_seen_mask, self._belief, belief_mean).astype(np.float32)
        seen_full = (self._global_seen_mask | self._global_obstacle_mask).astype(np.float32)
        obstacle_full = self._global_obstacle_mask.astype(np.float32)
        neighbor_full = np.zeros_like(seen_full, dtype=np.float32)  # single-agent test
        if self.use_recent_visit_map:
            self_recent_full = self._visit_age_to_recent(self._self_visit_age)
        else:
            self_recent_full = np.zeros_like(seen_full, dtype=np.float32)

        local_belief = self._crop(belief_vis_full, ci, cj)
        local_seen = self._crop(seen_full, ci, cj)
        local_obstacle = self._crop(obstacle_full, ci, cj)
        local_neighbor = self._crop(neighbor_full, ci, cj)
        local_recent = self._crop(self_recent_full, ci, cj)

        mask_seen = local_seen > 0.5
        vals = local_belief[mask_seen] if np.any(mask_seen) else local_belief
        q = max(float(np.quantile(vals - belief_mean, 0.99)), 0.0)
        delta_floor = belief_mean * (20.0 - 1.0)
        scale = max(q, delta_floor, 1e-6)
        b_scaled = np.clip((local_belief - belief_mean) / scale, 0.0, 1.0).astype(np.float32)

        img_base = np.stack([b_scaled, local_seen, local_obstacle, local_neighbor, local_recent], axis=0)
        img_base_u8 = (img_base * 255.0).clip(0, 255).astype(np.uint8)

        if len(self._img_history) == 0:
            for _ in range(self.frame_stack_k):
                self._img_history.append(img_base_u8.copy())
        else:
            self._img_history.append(img_base_u8.copy())

        hist_frames = list(self._img_history)
        if len(hist_frames) < self.frame_stack_k:
            pad_src = hist_frames[0]
            hist_frames = [pad_src.copy() for _ in range(self.frame_stack_k - len(hist_frames))] + hist_frames
        return np.concatenate(hist_frames[-self.frame_stack_k:], axis=0).astype(np.uint8, copy=False)

    def _norm_state(self, vx: float, vy: float, vz: float, yaw: float, wz: float, z_err: float, d: float, gx: float, gy: float) -> np.ndarray:
        vxy_max = max(1e-6, float(self.cfg.vxy_max))
        yaw_rate_max = max(1e-6, float(self.yaw_rate_max))
        safe_m = float(self.cfg.safe_margin_m)

        vz_cap = 1.0
        zerr_cap = 0.5
        d_cap = max(2.0, 6.0 * safe_m)
        g_cap = 2.0

        vx_n = np.clip(vx / vxy_max, -1.0, 1.0)
        vy_n = np.clip(vy / vxy_max, -1.0, 1.0)
        vz_n = np.clip(vz / vz_cap, -1.0, 1.0)
        sy, cy = np.sin(yaw), np.cos(yaw)
        wz_n = np.clip(wz / yaw_rate_max, -1.0, 1.0)
        zerr_n = np.clip(z_err / zerr_cap, -1.0, 1.0)

        if not np.isfinite(d):
            d = d_cap
        d_n = 2.0 * (np.clip(d, 0.0, d_cap) / d_cap) - 1.0

        if not np.isfinite(gx):
            gx = 0.0
        if not np.isfinite(gy):
            gy = 0.0
        gx_n = np.clip(gx, -g_cap, g_cap) / g_cap
        gy_n = np.clip(gy, -g_cap, g_cap) / g_cap

        return np.array([vx_n, vy_n, vz_n, sy, cy, wz_n, zerr_n, d_n, gx_n, gy_n], dtype=np.float32)

    def _norm_state_cbf(self, vx: float, vy: float, vz: float, yaw: float, wz: float, z_err: float, d: float, gx: float, gy: float) -> np.ndarray:
        vxy_max = max(1e-6, float(self.cfg.vxy_max))
        yaw_rate_max = max(1e-6, float(self.yaw_rate_max))
        safe_m = float(self.cfg.safe_margin_m)

        vz_cap = 1.0
        zerr_cap = 0.5
        d_cap = max(2.0, 6.0 * safe_m)
        g_cap = 2.0

        vx_n = np.clip(vx / vxy_max, -1.0, 1.0)
        vy_n = np.clip(vy / vxy_max, -1.0, 1.0)
        vz_n = np.clip(vz / vz_cap, -1.0, 1.0)
        sy, cy = np.sin(yaw), np.cos(yaw)
        wz_n = np.clip(wz / yaw_rate_max, -1.0, 1.0)
        zerr_n = np.clip(z_err / zerr_cap, -1.0, 1.0)

        if np.isfinite(d):
            d_n = 2.0 * (np.clip(d, 0.0, d_cap) / d_cap) - 1.0
        else:
            d_n = d
        gx_n = gx if not np.isfinite(gx) else np.clip(gx, -g_cap, g_cap) / g_cap
        gy_n = gy if not np.isfinite(gy) else np.clip(gy, -g_cap, g_cap) / g_cap

        return np.array([vx_n, vy_n, vz_n, sy, cy, wz_n, zerr_n, d_n, gx_n, gy_n], dtype=np.float32)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        x, y, z, yaw = self._get_pose_xyzyaw()
        vx, vy, vz = self._get_velocity()
        wz = self._get_wz()
        frame_bgr, frame_ts = self.frame_reader.get_latest()
        self._last_frame_ts = frame_ts

        self._update_real_maps_from_sensors(frame_bgr, x, y, yaw)
        self._advance_recent_visit(x, y)

        d, gx, gy = self._query_local_safety(x, y, z)
        qvis = self._compute_qvis(frame_bgr)
        qvio = self._compute_qvio(x, y, z, yaw)
        z_err = float(self.cfg.z_target - z)

        image = self._build_image_obs(x, y)
        state = self._norm_state(vx, vy, vz, yaw, wz, z_err, d, gx, gy)
        state_cbf = self._norm_state_cbf(vx, vy, vz, yaw, wz, z_err, d, gx, gy)
        if self.use_proxy_context:
            proxy = np.array([qvis, qvio], dtype=np.float32)
            state = np.concatenate([state, proxy], axis=0).astype(np.float32, copy=False)
            state_cbf = np.concatenate([state_cbf, proxy], axis=0).astype(np.float32, copy=False)

        return {
            "image": image[None, ...],
            "state": state[None, ...],
            "state_cbf": state_cbf[None, ...],
        }

    # ---------------------------------------------------------------------
    # Gym-like API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._global_seen_mask[:] = False
        self._global_obstacle_mask[:] = False
        self._belief[:] = 1.0 / float(max(1, self.map_h * self.map_w))
        self._self_visit_age.fill(self._recent_init_age)
        self._img_history.clear()
        self._prev_pose_for_vio_proxy = None
        self.depth_valid_ema = 1.0
        self.vio_consistency_ema = 1.0

        t0 = time.time()
        while self._pose_msg is None:
            if time.time() - t0 > 5.0:
                raise TimeoutError("Timed out waiting for MAVROS local position pose.")
            time.sleep(0.05)
        while self.frame_reader.get_latest()[0] is None:
            if time.time() - t0 > 5.0:
                raise TimeoutError("Timed out waiting for RTSP camera stream.")
            time.sleep(0.05)

        obs = self._build_obs()
        info = {
            "global_state": None,
            "real_backend": True,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(1, 3)
        a = action[0]
        ax = float(np.clip(a[0], -1.0, 1.0))
        ay = float(np.clip(a[1], -1.0, 1.0))
        az = float(np.clip(a[2], -1.0, 1.0))
        nxy = np.hypot(ax, ay)
        if nxy > 1.0:
            ax, ay = ax / nxy, ay / nxy

        vx_cmd = ax * float(self.cfg.vxy_max)
        vy_cmd = ay * float(self.cfg.vxy_max)
        yaw_rate_cmd = az * float(self.yaw_rate_max)

        self._publish_velocity_command(vx_cmd, vy_cmd, yaw_rate_cmd)
        time.sleep(self.dt_step)

        obs = self._build_obs()
        reward = 0.0  # inference-only minimal real env
        terminated = False
        truncated = False
        info = {
            "action_raw": (float(a[0]), float(a[1]), float(a[2])),
            "action_exec": (vx_cmd, vy_cmd, yaw_rate_cmd),
            "real_backend": True,
        }
        return obs, reward, terminated, truncated, info

    def _publish_velocity_command(self, vx: float, vy: float, yaw_rate: float) -> None:
        if self._cmd_pub is None:
            raise RuntimeError("ROS command publisher is not initialized.")
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(yaw_rate)
        self._cmd_pub.publish(msg)

    def render(self):
        frame, _ = self.frame_reader.get_latest()
        return frame

    def close(self):
        try:
            self.frame_reader.close()
        except Exception:
            pass


if __name__ == "__main__":
    cfg = RealEnvConfig(
        rtsp_url="rtsp://127.0.0.1:8900/live",
        command_topic="/mavros/setpoint_velocity/cmd_vel_unstamped",
    )
    env = RealDroneEnvMinimal(cfg)
    obs, info = env.reset()
    print("obs image shape:", obs["image"].shape)
    print("obs state shape:", obs["state"].shape)
    for _ in range(10):
        action = np.zeros((1, 3), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
    env.close()
