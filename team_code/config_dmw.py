import numpy as np


class GlobalConfig:
    def __init__(self):
        """ base architecture configurations """
        self.eval_route_as = 'target_point'  # "target_point", "target_point_command", or "command"
        self.use_cot = False

        self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
        self.carla_fps = 20  # Simulator frames per second
        self.stuck_threshold = 100
        self.creep_duration = 15  # Number of frames we will creep forward
        self.creep_throttle = 0.4
        self.inital_frames_delay = 2.0 / self.carla_frame_rate
        self.wp_dilation = 1  # Factor by which the wp are dilated compared to full CARLA 20 FPS
        self.data_save_freq = 5

        self.max_throttle = 1  # upper limit on throttle signal value in dataset
        self.brake_speed = 0.4  # desired speed below which brake is triggered
        self.brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
        self.clip_delta = 1.0  # maximum change in speed input to longitudinal controller
        self.clip_throttle = 1.0  # maximum throttle allowed by the controller

        # -----------------------------------------------------------------------------
        # PID controller
        # -----------------------------------------------------------------------------
        # We are minimizing the angle to the waypoint that is at least aim_distance
        # meters away, while driving
        self.aim_distance_very_fast = 7.0
        self.aim_distance_fast = 3.0
        self.aim_distance_slow = 2.25
        # Meters per second threshold switching between aim_distance_fast and aim_distance_slow
        self.aim_distance_threshold = 5.5
        self.aim_distance_threshold2 = 15
        self.turn_kp = 3.25
        self.turn_ki = 1.0
        self.turn_kd = 1.0
        self.turn_n = 20  # buffer size

        self.speed_kp = 1.75
        self.speed_ki = 1.0
        self.speed_kd = 2.0
        self.speed_n = 20  # buffer size

        # -----------------------------------------------------------------------------
        # Sensor config
        # -----------------------------------------------------------------------------
        self.num_cameras = [0]
        self.camera_pos_0 = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
        self.camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

        self.camera_width_0 = 1024  # Camera width in pixel during data collection
        self.camera_height_0 = 512  # Camera height in pixel during data collection
        self.camera_fov_0 = 110

        self.detection_radius = 50.0
        self.high_speed_min_extent_x_other_vehicle = 1.2
        self.slow_speed_extent_factor_ego = 1.0
        self.extent_other_vehicles_bbs_speed_threshold = 1.0
        self.high_speed_min_extent_y_other_vehicle = 1.0
        self.high_speed_extent_y_factor_other_vehicle = 1.3

        self.bicycle_frame_rate = 20
        self.forecast_length_lane_change = 1.1
        self.default_forecast_length = 2.0

        # -----------------------------------------------------------------------------
        # Kinematic Bicycle Model
        # -----------------------------------------------------------------------------
        # Time step for the model (20 frames per second).
        self.time_step = 1. / 20.
        # Kinematic bicycle model parameters tuned from World on Rails.
        # Distance from the rear axle to the front axle of the vehicle.
        self.front_wheel_base = -0.090769015
        # Distance from the rear axle to the center of the rear wheels.
        self.rear_wheel_base = 1.4178275
        # Gain factor for steering angle to wheel angle conversion.
        self.steering_gain = 0.36848336
        # Deceleration rate when braking (m/s^2) of other vehicles.
        self.brake_acceleration = -4.952399
        # Acceleration rate when throttling (m/s^2) of other vehicles.
        self.throttle_acceleration = 0.5633837
        # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
        # Tuned using a dataset where the car drives on a straight highway, accelerates, and brakes.
        self.throttle_values = np.array([
            9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
            -1.05996152e-03, 6.71079346e-04
        ])
        # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
        self.brake_values = np.array([
            9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
            -4.91381935e-12
        ])
        # Minimum throttle value that has an effect during forecasting the ego vehicle.
        self.throttle_threshold_during_forecasting = 0.3
