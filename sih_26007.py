import math
import json
import random
import time
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu, NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class AdaptiveEKF:
    """
    6-DOF Adaptive Extended Kalman Filter (AEKF) for Mining Dumper Localization.
    State vector x = [px, py, pz, v, yaw, yaw_rate]^T
    Fuses IMU, GPS, Lidar, and 77 GHz mmWave Radar.
    Uses Sage-Husa innovation covariance matching + weather-informed noise scaling.
    """
    def __init__(self, truck_id, init_state, init_z=0.45, init_speed=0.45):
        self.truck_id = truck_id
        self.x = np.array([
            float(init_state[0]),
            float(init_state[1]),
            float(init_z),
            float(init_speed),
            float(init_state[2]),
            0.0
        ], dtype=np.float64)

        self.P = np.diag([0.25, 0.25, 0.15, 0.10, 0.04, 0.02]).astype(np.float64)
        self.Q_base = np.diag([0.008, 0.008, 0.004, 0.015, 0.004, 0.008]).astype(np.float64)
        self.Q = self.Q_base.copy()

        # Measurement vector z = [gps_x, gps_y, gps_z, lidar_x, lidar_y, lidar_yaw, radar_v, imu_yaw, imu_yaw_rate]^T
        self.R_base = np.diag([
            0.35**2, 0.35**2, 0.40**2,   # GPS x, y, z
            0.08**2, 0.08**2, 0.03**2,   # Lidar x, y, yaw
            0.06**2,                     # Radar Doppler ego-velocity
            0.025**2, 0.015**2           # IMU yaw, yaw_rate
        ]).astype(np.float64)
        self.R = self.R_base.copy()
        self.C_innov = self.R_base.copy()

        self.H = np.zeros((9, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # gps_x
        self.H[1, 1] = 1.0  # gps_y
        self.H[2, 2] = 1.0  # gps_z
        self.H[3, 0] = 1.0  # lidar_x
        self.H[4, 1] = 1.0  # lidar_y
        self.H[5, 4] = 1.0  # lidar_yaw
        self.H[6, 3] = 1.0  # radar_v
        self.H[7, 4] = 1.0  # imu_yaw
        self.H[8, 5] = 1.0  # imu_yaw_rate

        self.innovation_norm = 0.0
        self.sensor_weights = {"GPS": 0.25, "LIDAR": 0.35, "RADAR": 0.20, "IMU": 0.20}
        self.history = []
        self.gps_raw_history = []

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def reset(self, init_state, init_z=0.45, init_speed=0.45):
        self.x = np.array([
            float(init_state[0]),
            float(init_state[1]),
            float(init_z),
            float(init_speed),
            float(init_state[2]),
            0.0
        ], dtype=np.float64)
        self.P = np.diag([0.25, 0.25, 0.15, 0.10, 0.04, 0.02]).astype(np.float64)
        self.C_innov = self.R_base.copy()
        self.history.clear()
        self.gps_raw_history.clear()

    def predict(self, dt, accel_lon, yaw_rate_meas, pitch=0.0):
        px, py, pz, v, yaw, yr = self.x
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # Nonlinear state transition f(x, u)
        self.x[0] = px + v * cos_y * cos_p * dt
        self.x[1] = py + v * sin_y * cos_p * dt
        self.x[2] = pz - v * sin_p * dt
        self.x[3] = max(0.0, v + accel_lon * dt)
        self.x[4] = self._wrap_angle(yaw + (0.5 * yr + 0.5 * yaw_rate_meas) * dt)
        self.x[5] = 0.7 * yr + 0.3 * yaw_rate_meas

        # Jacobian F = df/dx
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = cos_y * cos_p * dt
        F[0, 4] = -v * sin_y * cos_p * dt
        F[1, 3] = sin_y * cos_p * dt
        F[1, 4] = v * cos_y * cos_p * dt
        F[2, 3] = -sin_p * dt
        F[4, 5] = 0.5 * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_meas, fog_density, rain_intensity, canyon_factor=1.0):
        # Weather-informed prior scaling on R
        gps_scale = (1.0 + 1.8 * rain_intensity + 0.5 * fog_density) * canyon_factor
        lidar_scale = 1.0 + 5.5 * fog_density + 3.8 * rain_intensity
        radar_scale = 1.0 + 0.12 * fog_density + 0.45 * rain_intensity
        imu_scale = 1.0 + 0.25 * rain_intensity

        R_weather = self.R_base.copy()
        R_weather[0:3, 0:3] *= (gps_scale ** 2)
        R_weather[3:6, 3:6] *= (lidar_scale ** 2)
        R_weather[6, 6] *= (radar_scale ** 2)
        R_weather[7:9, 7:9] *= (imu_scale ** 2)

        # Innovation y = z - H*x
        z_pred = self.H @ self.x
        y = z_meas - z_pred
        y[5] = self._wrap_angle(y[5])
        y[7] = self._wrap_angle(y[7])

        # Sage-Husa adaptive innovation covariance matching
        alpha_sh = 0.15
        self.C_innov = (1.0 - alpha_sh) * self.C_innov + alpha_sh * np.outer(y, y)
        R_emp = np.diag(np.clip(np.diag(self.C_innov - self.H @ self.P @ self.H.T), 1e-4, 25.0))
        self.R = 0.65 * R_weather + 0.35 * R_emp

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        dx = K @ y
        self.x = self.x + dx
        self.x[3] = max(0.0, self.x[3])
        self.x[4] = self._wrap_angle(self.x[4])

        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Adaptive process noise Q tuning via innovation magnitude
        self.Q = 0.85 * self.Q_base + 0.15 * np.diag(np.clip(np.diag(K @ self.C_innov @ K.T), 1e-5, 0.2))
        self.innovation_norm = float(np.linalg.norm(y[:5]))

        # Compute relative sensor trust weights from inverse measurement variance
        inv_gps = 1.0 / max(1e-4, float(np.mean(np.diag(self.R)[0:2])))
        inv_lid = 1.0 / max(1e-4, float(np.mean(np.diag(self.R)[3:5])))
        inv_rad = 1.0 / max(1e-4, float(self.R[6, 6]))
        inv_imu = 1.0 / max(1e-4, float(np.mean(np.diag(self.R)[7:9])))
        total_inv = inv_gps + inv_lid + inv_rad + inv_imu
        self.sensor_weights = {
            "GPS": inv_gps / total_inv,
            "LIDAR": inv_lid / total_inv,
            "RADAR": inv_rad / total_inv,
            "IMU": inv_imu / total_inv
        }

        self.history.append((float(self.x[0]), float(self.x[1])))
        if len(self.history) > 90:
            self.history.pop(0)

        self.gps_raw_history.append((float(z_meas[0]), float(z_meas[1])))
        if len(self.gps_raw_history) > 45:
            self.gps_raw_history.pop(0)

        return self.x.copy()


class BailadilaProductionEngine(Node):
    def __init__(self):
        super().__init__('bailadila_production_engine')

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub_markers = self.create_publisher(MarkerArray, '/mining_scene_markers', 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.bridge = CvBridge()

        # Sensor Suite Publishers (Camera, Lidar, Radar, IMU, GPS)
        self.pub_cams = {i: self.create_publisher(Image, f'/dumper_{i}/camera/image_raw', sensor_qos) for i in range(1, 7)}
        self.pub_lidar = {i: self.create_publisher(PointCloud2, f'/dumper_{i}/lidar/points', sensor_qos) for i in range(1, 7)}
        self.pub_radar = {i: self.create_publisher(PointCloud2, f'/dumper_{i}/radar/detections', sensor_qos) for i in range(1, 7)}
        self.pub_imu = {i: self.create_publisher(Imu, f'/dumper_{i}/imu/data', sensor_qos) for i in range(1, 7)}
        self.pub_gps = {i: self.create_publisher(NavSatFix, f'/dumper_{i}/gps/fix', sensor_qos) for i in range(1, 7)}

        # Decentralized V2V Publishers & Subscribers
        self.pub_v2v = {i: self.create_publisher(String, f'/v2v/dumper_{i}/bsm', sensor_qos) for i in range(1, 7)}
        self.pub_v2v_bus = self.create_publisher(String, '/fleet/v2v_bus', sensor_qos)
        self.v2v_inbox = {i: {} for i in range(1, 7)}
        self.v2v_subs = []
        for rx_id in range(1, 7):
            for tx_id in range(1, 7):
                if rx_id != tx_id:
                    sub = self.create_subscription(
                        String,
                        f'/v2v/dumper_{tx_id}/bsm',
                        self._create_v2v_callback(rx_id),
                        sensor_qos
                    )
                    self.v2v_subs.append(sub)

        # 3-Screen Display Publishers + Combined Cockpit Monitor
        self.pub_cockpit = self.create_publisher(Image, '/fleet/cockpit_monitor', sensor_qos)
        self.pub_screen_1 = self.create_publisher(Image, '/fleet/screen_1_pov', sensor_qos)
        self.pub_screen_2 = self.create_publisher(Image, '/fleet/screen_2_bev', sensor_qos)
        self.pub_screen_3 = self.create_publisher(Image, '/fleet/screen_3_telemetry', sensor_qos)

        self.w, self.h = 320, 220
        self.horizon_y = 105
        self.lane_right_x = -7.5
        self.lane_left_x = -11.5
        self.blue_lane_x = 0.0
        self.theta = 0.139626
        self.cos_t = math.cos(self.theta)

        self.sal_trees = [
            (-18.0, -22.0), (-18.0, 36.0), (12.0, -18.0), (14.0, 28.0),
            (-22.0, -5.0), (-22.0, 18.0), (26.0, -12.0), (28.0, 20.0),
            (-32.0, 2.0), (-32.0, 25.0), (32.0, -25.0), (35.0, 10.0), (14.0, 42.0), (-18.0, 48.0)
        ]

        # Static obstacles (x_center, y_center, half_x, half_y, z_height, reflectivity) for raycasting
        self.static_aabbs = [
            (-26.0, 15.0, 3.5, 75.0, 5.0, 140.0),   # West Highwall 1
            (-33.0, 15.0, 3.5, 75.0, 7.0, 130.0),   # West Highwall 2
            (40.0, 15.0, 4.5, 75.0, 7.0, 140.0),    # East Highwall
            (4.8, 7.8, 2.5, 2.5, 3.8, 165.0),       # North Island
            (4.8, -7.8, 2.5, 2.5, 3.8, 165.0),      # South Island
            (-14.8, 12.0, 0.65, 55.0, 2.2, 120.0),  # Left Safety Berm
            (-4.2, -8.0, 0.60, 30.0, 2.2, 120.0),   # Right Safety Berm
        ]

        random.seed(42)
        self.dust_particles = []
        for _ in range(24):
            self.dust_particles.append({
                'truck': random.choice([3, 4, 6]),
                'dx': random.uniform(-3.5, -1.2),
                'dy': random.uniform(-0.8, 0.8),
                'dz': random.uniform(0.1, 1.2),
                'size': random.uniform(0.8, 1.8),
                'alpha': random.uniform(0.25, 0.55)
            })

        self.rain_drops = [
            {
                'x': random.uniform(0, 480),
                'y': random.uniform(0, 320),
                'len': random.uniform(8.0, 20.0),
                'speed': random.uniform(14.0, 26.0),
                'layer': random.choice([0, 1, 2])
            }
            for _ in range(140)
        ]

        self.lock = threading.Lock()
        self.reset_simulation()

        self.timer = self.create_timer(0.04, self.tick)
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()

        self.get_logger().info("Bailadila Engine Active: Fog/Rain + Lidar/Radar + AEKF + V2V + 3-Screen Display Ready.")

    def _create_v2v_callback(self, receiver_id):
        def callback(msg):
            try:
                pkt = json.loads(msg.data)
                sender_id = int(pkt.get("truck_id", -1))
                if sender_id < 1 or sender_id == receiver_id:
                    return
                rx_pos = self.aekf[receiver_id].x[:2] if hasattr(self, 'aekf') and receiver_id in self.aekf else self.state[receiver_id][:2]
                tx_pos = (float(pkt["x"]), float(pkt["y"]))
                dist = math.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1])

                path_loss = 42.0 + 22.0 * math.log10(max(1.0, dist))
                nlos_penalty = 8.0 if (tx_pos[0] * rx_pos[0] < 0 and abs(tx_pos[1] - rx_pos[1]) > 10.0) else 0.0
                weather_att = 3.5 * self.rain_intensity + 1.2 * self.fog_density
                rssi_dbm = -20.0 - path_loss - nlos_penalty - weather_att + random.gauss(0.0, 0.8)

                if rssi_dbm > -98.0:
                    pkt["rssi_dbm"] = round(rssi_dbm, 1)
                    pkt["rx_dist"] = round(dist, 2)
                    pkt["rx_stamp"] = self.sim_time
                    self.v2v_inbox[receiver_id][sender_id] = pkt
            except Exception:
                pass
        return callback

    def reset_simulation(self):
        self.sim_time = 0.0
        self.fog_density = 0.12
        self.rain_intensity = 0.05
        self.road_friction = 0.78
        self.weather_label = "CLEAR / LIGHT MIST"

        self.state = {
            1: [0.0, -18.0, 1.5708],   # Orange
            2: [18.0, 2.0, 3.14159],   # Blue
            3: [-7.5, -8.0, 1.5708],   # Yellow
            4: [-7.5, -18.0, 1.5708],  # Green Ego
            5: [0.0, 22.0, -1.5708],   # Red
            6: [-11.5, 14.0, -1.5708]  # Purple
        }
        self.speeds = {1: 0.45, 2: 0.50, 3: 0.22, 4: 0.58, 5: 0.55, 6: 0.48}
        self.prev_speeds = self.speeds.copy()
        self.prev_yaws = {i: self.state[i][2] for i in range(1, 7)}
        self.braking = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False}
        self.turn_signal = {1: "NONE", 2: "NONE", 3: "NONE", 4: "NONE", 5: "NONE", 6: "NONE"}
        self.payload_status = {
            1: "HAULING_ORE", 2: "EMPTY_RETURNING", 3: "HAULING_ORE",
            4: "HAULING_ORE", 5: "EMPTY_RETURNING", 6: "EMPTY_RETURNING"
        }
        self.intent_phase = {
            1: "NORTHBOUND", 2: "APPROACHING_STOP_BAY", 3: "RAMP_CLIMB_SLOW",
            4: "APPROACH", 5: "APPROACHING_NORTH_BAY", 6: "RAMP_DESCENT"
        }

        self.t1_phase = "NORTHBOUND"
        self.t1_arc = 0.0
        self.t1_cleared = False
        self.t5_phase = "APPROACHING_NORTH_BAY"
        self.t2_phase = "APPROACHING_STOP_BAY"
        self.t2_arc = 0.0
        self.t4_phase = "APPROACH"
        self.t4_progress = 0.0
        self.t4_merge_progress = 0.0

        self.aekf = {
            i: AdaptiveEKF(i, self.state[i], self.calc_z(i), self.speeds[i])
            for i in range(1, 7)
        }
        self.lidar_cache = {i: [] for i in range(1, 7)}
        self.radar_detections = {i: [] for i in range(1, 7)}
        self.algorithm_results = {
            i: {
                "closest_dist": 999.0,
                "closest_id": None,
                "ttc": 99.9,
                "stop_dist": 2.5,
                "risk_level": "NOMINAL",
                "targets": []
            }
            for i in range(1, 7)
        }
        self.v2v_inbox = {i: {} for i in range(1, 7)}

    def update_weather_dynamics(self):
        phase_angle = (self.sim_time / 48.0) * 2.0 * math.pi
        self.fog_density = float(np.clip(0.38 + 0.34 * math.sin(phase_angle - 0.6), 0.06, 0.76))
        self.rain_intensity = float(np.clip(0.36 + 0.42 * math.sin(phase_angle - 1.1), 0.0, 0.82))
        self.road_friction = float(np.clip(0.80 - 0.36 * self.rain_intensity - 0.08 * self.fog_density, 0.36, 0.80))

        if self.rain_intensity > 0.55 and self.fog_density > 0.50:
            self.weather_label = "MONSOON SQUALL & DENSE FOG"
        elif self.rain_intensity > 0.30:
            self.weather_label = "MODERATE RAIN & MIST"
        elif self.fog_density > 0.45:
            self.weather_label = "HEAVY HILL FOG"
        else:
            self.weather_label = "LIGHT MIST / CLEAR"

    def create_header(self, frame_id="odom"):
        h = Header()
        h.frame_id = frame_id
        h.stamp = self.get_clock().now().to_msg()
        return h

    def calc_z(self, truck_id):
        curr_y = self.state[truck_id][1]
        if truck_id in [3, 4, 6]:
            ramp_z = 1.2 - (curr_y - 12.0) * math.tan(self.theta) + 0.45
            if truck_id in [3, 4] and curr_y > 16.0:
                blend = min(1.0, max(0.0, (curr_y - 16.0) / 12.0))
                return (1.0 - blend) * ramp_z + blend * 0.45
            return ramp_z
        return 0.45

    def _raycast_lidar_pointcloud(self, truck_id):
        ego_x, ego_y, ego_yaw = self.state[truck_id]
        ego_z = self.calc_z(truck_id) + 1.2
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)

        max_range = 42.0 * (1.0 - 0.45 * self.fog_density - 0.25 * self.rain_intensity)
        sigma_r = 0.025 + 0.28 * self.fog_density + 0.20 * self.rain_intensity
        dropout_prob = 0.02 + 0.18 * self.rain_intensity + 0.12 * self.fog_density

        azimuth_steps = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        elevation_channels = np.radians([-15.0, -10.0, -6.0, -2.5, 0.0, 3.5, 7.0, 12.0])

        cylinders = []
        for other_id in range(1, 7):
            if other_id == truck_id:
                continue
            ox, oy, _ = self.state[other_id]
            oz = self.calc_z(other_id)
            cylinders.append((ox, oy, 1.55, oz + 1.4, 245.0))

        for tx, ty in self.sal_trees:
            if abs(tx - ego_x) < 38.0 and abs(ty - ego_y) < 38.0:
                cylinders.append((tx, ty, 0.85, 4.8, 95.0))

        points = []
        bev_hits_world = []

        for az in azimuth_steps:
            if random.random() < (0.14 * self.fog_density + 0.18 * self.rain_intensity):
                clutter_d = random.uniform(1.6, 6.8)
                clutter_el = random.uniform(-0.15, 0.18)
                cx = clutter_d * math.cos(clutter_el) * math.cos(az)
                cy = clutter_d * math.cos(clutter_el) * math.sin(az)
                cz = clutter_d * math.sin(clutter_el)
                points.append([float(cx), float(cy), float(cz), float(random.uniform(18.0, 48.0))])

            world_az = ego_yaw + az
            dir_wx = math.cos(world_az)
            dir_wy = math.sin(world_az)

            for el in elevation_channels:
                if random.random() < dropout_prob:
                    continue

                cos_el = math.cos(el)
                sin_el = math.sin(el)
                best_t = max_range
                best_intensity = 0.0
                hit_found = False

                if sin_el < -0.02:
                    t_ground = -1.22 / sin_el
                    if 2.0 < t_ground < best_t:
                        best_t = t_ground
                        best_intensity = 65.0
                        hit_found = True

                for cx_w, cy_w, rad, top_z, refl in cylinders:
                    dx = cx_w - ego_x
                    dy = cy_w - ego_y
                    proj = dx * dir_wx + dy * dir_wy
                    if proj <= 0.0 or proj > best_t * cos_el:
                        continue
                    perp_sq = (dx * dx + dy * dy) - (proj * proj)
                    if perp_sq < rad * rad:
                        t_horiz = max(1.2, proj - math.sqrt(rad * rad - perp_sq))
                        t_3d = t_horiz / max(0.1, cos_el)
                        hit_z = ego_z + t_3d * sin_el
                        if -0.3 <= hit_z <= top_z and t_3d < best_t:
                            best_t = t_3d
                            best_intensity = refl
                            hit_found = True

                for bx, by, hx, hy, hz, refl in self.static_aabbs:
                    min_x, max_x = bx - hx, bx + hx
                    min_y, max_y = by - hy, by + hy
                    t_min, t_max = 1.0, best_t * cos_el

                    if abs(dir_wx) > 1e-5:
                        tx1 = (min_x - ego_x) / dir_wx
                        tx2 = (max_x - ego_x) / dir_wx
                        t_min = max(t_min, min(tx1, tx2))
                        t_max = min(t_max, max(tx1, tx2))
                    elif not (min_x <= ego_x <= max_x):
                        continue

                    if abs(dir_wy) > 1e-5:
                        ty1 = (min_y - ego_y) / dir_wy
                        ty2 = (max_y - ego_y)