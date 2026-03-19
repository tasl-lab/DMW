"""
partially taken from https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/sensor_agent.py
(MIT licence)
"""


import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import carla
import cv2
import hydra
import numpy as np
import torch
import ujson
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from hydra.utils import get_original_cwd, to_absolute_path
from leaderboard.autoagents import autonomous_agent
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve
from transformers import AutoConfig, AutoProcessor

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import scenario_logger
import team_code.transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from models.utils.custom_types import DrivingInput, LanguageLabel
from models.utils.internvl2_utils import build_transform, dynamic_preprocess
from team_code.config_dmw import GlobalConfig
from team_code.nav_planner import LateralPIDController, RoutePlanner
from team_code.dmw_utils import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    get_rotation_matrix,
    project_points,
)
from team_code.kinematic_bicycle_model import KinematicBicycleModel
# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

import json
import random


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return 'LongTermPreferenceAgent'


DEBUG = True
HD_VIZ = False
USE_UKF = True

class LongTermPreferenceAgent(autonomous_agent.AutonomousAgent):
    """
        Main class that runs the agents with the run_step function
        """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None, scenario_configs=None):
        """Sets up the agent. route_index is for logging purposes"""

        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.SENSORS
        if '+' in path_to_conf_file:
            print(f"path to conf file: {path_to_conf_file}")
            self.config_path = path_to_conf_file.split('+')[0]
            print(f"Config path: {self.config_path}")
            self.save_path_root = path_to_conf_file.split('+')[1]
            print(f"Save path root: {self.save_path_root}")
        else:
            self.config_path = path_to_conf_file
            print(f"Config path: {self.config_path}")
            self.save_path_root = route_index
            print(f"Save path root: {self.save_path_root}")
        self.step = -1
        self.initialized = False
        self.device = torch.device('cuda')
        self.DrivingInput = {}
        self.config = GlobalConfig()

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = self.model.route_as

        self.scenario_configs = scenario_configs
        for scenario_config in self.scenario_configs:
            print(f"Scenario config: {scenario_config.type}")

        self.last_command = -1
        self.last_command_tmp = -1
        self.user_command = None
        self.user_flag = None
        self.running = True
        self.custom_prompt = None
        
        self.LMDRIVE_AUGM = False
        if self.LMDRIVE_AUGM:
                command_templates_file = f"data/augmented_templates/lmdrive.json"
                with open(command_templates_file, 'r') as f:
                        self.command_templates = ujson.load(f)
        
        self.speed_controller = t_u.PIDController(k_p=self.config.speed_kp,
                                                                                            k_i=self.config.speed_ki,
                                                                                            k_d=self.config.speed_kd,
                                                                                            n=self.config.speed_n)

        self.turn_controller = LateralPIDController(inference_mode=False)

        image_fps = 5
        image_history_length = 1

        self.image_buffer = deque(maxlen=image_fps * image_history_length)

        self.lidar_seq_len = 1
        self.logging_freq = 10  # Log every 10 th frame
        self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
        self.dense_route_planner_min_distance = 1.0
        self.dense_route_planner_max_distance = 50.0
        self.log_route_planner_min_distance = 4.0
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        #load config from .hydra folder
        self.config_load_path = Path(self.config_path).parent.parent / '.hydra' / 'config_grpo_human.yaml'
        with open(self.config_load_path, 'r') as file:
            cfg = OmegaConf.load(file)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    
        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        if 'tokenizer' in processor.__dict__:
                self.tokenizer = processor.tokenizer
        else:
                self.tokenizer = processor
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"
        cache_dir = f"pretrained/{(cfg.model.vision_model.variant.split('/')[1])}"
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=cache_dir,
                _recursive_=False
            ).to(self.device)
        torch.set_default_dtype(default_dtype)
        # Load the GRPO checkpoint (saved as pytorch_model.pt)
        self.model.load_state_dict(torch.load(self.config_path, map_location=self.device), strict=False)

        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Filtering
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(dim_x=4,
                                        dim_z=4,
                                        fx=bicycle_model_forward,
                                        hx=measurement_function_hx,
                                        dt=self.config.carla_frame_rate,
                                        points=self.points,
                                        x_mean_fn=state_mean,
                                        z_mean_fn=measurement_mean,
                                        residual_x=residual_state_x,
                                        residual_z=residual_measurement_h)

            # State noise, same as measurement because we
            # initialize with the first measurement later
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            # Measurement noise
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
            # Used to set the filter state equal the first measurement
            self.filter_initialized = False
        # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
        self.state_log = deque(maxlen=max((self.lidar_seq_len * self.config.data_save_freq), 2))

        self.datagen = int(os.environ.get("DATAGEN", 0)) == 1

        if os.environ.get("SAVE_PATH", None) is not None:
            string = os.environ["TOWN"]
            string += "_Rep" + os.environ["REPETITION"]
            string += f"_{route_index}"

            self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            if self.datagen:
                (self.save_path / "measurements").mkdir()

        # Logger that generates logs used for infraction replay in the results_parser.
        if self.save_path is not None and route_index is not None:
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

            self.lon_logger = ScenarioLogger(
                    save_path=self.save_path,
                    route_index=route_index,
                    logging_freq=self.logging_freq,
                    log_only=True,
                    route_only=False,  # with vehicles
                    roi=self.logger_region_of_interest,
            )
        
        self.debug_save_path = self.save_path
        Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        # self.save_path_metric = self.debug_save_path + '/metric'
        self.save_path_metric = self.save_path / 'metric'
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        if DEBUG:
            # self.save_path_img = self.debug_save_path + '/images'
            self.save_path_img = self.save_path / 'images'
            Path(self.save_path_img).mkdir(parents=True, exist_ok=True)

        self.pred_speed_wps = None
        self.pred_route = None

        self.prompt_type = None
        self.driver_id = None


    def input_thread(self):
        while self.running:
            user_input = input("Enter a command for the vehicle. 1: turn left, 2: turn right, 3: lane change left, 4: lane change right, 5: stop, 6: accelerate: ")
            if user_input.isdigit():
                    self.user_flag = int(user_input)
            else:
                self.user_command = str(user_input)
                
            if user_input.strip().lower() == "exit":
                self.running = False
            
            print(f"User command: {self.user_command}")
            print(f"User flag: {self.user_flag}")

    def _init(self):
        # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
        # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
        # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
        # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
        # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
        # of the GPS. It seems to work reasonably well, so we use this workaround for now.
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation
            def equations(variables):
                x, y = variables
                eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                             - math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
                             * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
                             * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]
            initial_guess = [0.0, 0.0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0
        self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
                                                                             self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self.vehicle_model = KinematicBicycleModel(self.config)
        self.ego_model = KinematicBicycleModel(self.config)

        self.lane_change_count = 0
        self.previous_lane_id = None

        with open(os.path.join(os.environ.get("WORK_DIR", ""), 'user_profile_data', f'personal_info/{self.driver_id}.json'), "r") as f:
            self.user_profile = json.load(f)

        world_map = self._world.get_map()

    def sensors(self):
        sensors = []
        for num_cam in self.config.num_cameras:
            # get from config by name as string
            sensors += [
                    {
                            'type': 'sensor.camera.rgb',
                            'x': self.config.__dict__[f'camera_pos_{num_cam}'][0],
                            'y': self.config.__dict__[f'camera_pos_{num_cam}'][1],
                            'z': self.config.__dict__[f'camera_pos_{num_cam}'][2],
                            'roll': self.config.__dict__[f'camera_rot_{num_cam}'][0],
                            'pitch': self.config.__dict__[f'camera_rot_{num_cam}'][1],
                            'yaw': self.config.__dict__[f'camera_rot_{num_cam}'][2],
                            'width': self.config.__dict__[f'camera_width_{num_cam}'],
                            'height': self.config.__dict__[f'camera_height_{num_cam}'],
                            'fov': self.config.__dict__[f'camera_fov_{num_cam}'],
                            'id': f'rgb_{num_cam}'
                    }
            ]

        if HD_VIZ:
            sensors += [{
                                                'type': 'sensor.camera.rgb',
                                                'x': -5.5, 'y': 0.0, 'z':3.5,
                                                'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                                                'width': 1920, 'height': 1080, 'fov': 110,
                                                'id': 'rgb_viz'
            }]

        sensors += [{
                'type': 'sensor.other.imu',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': self.config.carla_frame_rate,
                'id': 'imu'
        }, {
                'type': 'sensor.other.gnss',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
        }, {
                'type': 'sensor.speedometer',
                'reading_frequency': self.config.carla_fps,
                'id': 'speed'
        }, 
        ]

        return sensors

    @torch.inference_mode()  # Turns off gradient computation
    def tick(self, input_data):
        """Pre-processes sensor data and runs the Unscented Kalman Filter"""
        rgb = []

        if HD_VIZ:
            self.hd_cam_for_viz = input_data['rgb_viz'][1][:, :, :3]

        for camera_pos in self.config.num_cameras:
            rgb_cam = 'rgb_' + str(camera_pos)
            camera = input_data[rgb_cam][1][:, :, :3]
            if camera_pos == 0:
                self.camera_for_viz = camera.copy()

            # Also add jpg artifacts at test time, because the training data was saved as jpg.
            _, compressed_image_i = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

            rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_pos = rgb_pos[:int(rgb_pos.shape[0] - (rgb_pos.shape[0] * 4.8) // 16), :, :] # do this from config to ensure it is the same as in training

            # Switch to pytorch channel first order
            rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
            rgb.append(rgb_pos)

        rgb = np.array(rgb)
        self.image_buffer.append(rgb)

        rgbs = rgb
        image_sizes = None
        
        if 'internvl2' in self.cfg.model.vision_model.variant.lower():
            T, C, H, W = rgbs.shape
            transform = build_transform(input_size=448)
            images_processed_tmp = []
            images_sizes_tmp = []
            
            image = Image.fromarray(rgbs.squeeze(0).transpose(1, 2, 0))
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=self.cfg.model.vision_model.use_global_img, max_num=2)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            images_processed_tmp.append(pixel_values)
            images_sizes_tmp.append([image.size[1], image.size[0]])
            
            images_processed = {
                    'pixel_values': torch.stack(images_processed_tmp), 
                    'image_sizes': torch.tensor(images_sizes_tmp)
                    }  
            processed_image = images_processed['pixel_values']
            num_patches = processed_image.shape[1]
            new_height = processed_image.shape[3]
            new_width = processed_image.shape[4]
            processed_image = processed_image.view(1, self.T, num_patches, C, new_height, new_width)
            
        else:
            raise NotImplementedError(f"Encoder {self.cfg.data_module.encoder} not implemented yet")
        
        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result = {
                'rgb': rgb,
                'compass': compass,
        }
        speed = input_data['speed'][1]['speed']

        if USE_UKF:
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                self.filter_initialized = True

            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
            filtered_state = self.ukf.x

            self.state_log.append(filtered_state)
            result['gps'] = filtered_state[0:2]
        else:
            result['gps'] = np.array([gps_pos[0], gps_pos[1]])
            
        speed = round(input_data['speed'][1]['speed'], 1)

        waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))

        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            next_target_point, next_far_command = waypoint_route[1]
        else:
            target_point, far_command = waypoint_route[0]
            next_target_point, next_far_command = waypoint_route[0]
            
            
        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        
        self.last_command_tmp = far_command
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)

        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

        result['target_point'] = ego_target_point_torch

        self.target_points = None
        placeholder_batch_list = []

        if self.config.eval_route_as == 'target_point' or self.config.eval_route_as == 'target_point_command':
            target_points = [ego_target_point, ego_next_target_point]
            self.target_points = target_points.copy()
            target_points_np = np.array(target_points)
            target_points = torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)
            result['route'] = target_points
            
            placeholder_values = {'<TARGET_POINT>': target_points_np}
            tmp = {}
            for key, value in placeholder_values.items():
                    token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                    tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
            
            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
            
        elif self.config.eval_route_as == 'command':
            # get distance from target_point
            dist_to_command = np.linalg.norm(ego_target_point)
            dist_to_command = int(dist_to_command)
            map_command = {
                    1: 'go left at the next intersection',
                    2: 'go right at the next intersection',
                    3: 'go straight at the next intersection',
                    4: 'follow the road',
                    5: 'do a lane change to the left',
                    6: 'do a lane change to the right',        
            }
            command_template_mappings = {
                    1: [0, 2, 4, 7],
                    2: [1, 3, 5, 8],
                    3: [6, 9],
                    4: [38, 40, 42, 43, 44, 45],
                    5: [34, 36],
                    6: [35, 37],
            }
            if self.LMDRIVE_AUGM:
                lmdrive_index = random.choice(command_template_mappings[far_command])
                lmdrive_command = random.choice(self.command_templates[str(lmdrive_index)])
                lmdrive_command = lmdrive_command.replace('[x]', str(dist_to_command))
                prompt_tp = f'Command: {lmdrive_command}'
                
            else:
                command = map_command[far_command]
                next_command = map_command[next_far_command]
                if self.last_command in [1, 2, 3] and far_command == 4:
                    next_command = command
                    command = map_command[self.last_command]
                    
                if command != next_command:
                        next_command = f' then {next_command}'
                else:
                        next_command = ''
                        
                if far_command == 4:
                        prompt_tp = f'Command: {command}{next_command}.'
                else:
                        prompt_tp = f'Command: {command} in {dist_to_command} meter{next_command}.'
                
        else:
            result['route'] = route_img


        if self.config.use_cot:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."
        
        if self.custom_prompt is not None:
            if self.user_flag == 2 or self.user_flag == 3:
                prompt = f"Current speed: {speed} m/s. {self.custom_prompt}"
            else:
                prompt = f"Current speed: {speed} m/s. {prompt_tp} {self.custom_prompt}"


        if self.user_flag == 1 or self.user_flag == 2:
            prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"
        elif self.user_flag == 0:
            prompt = f"<SAFETY> {prompt}"


        result['speed'] = torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)

        B, T, num_patches, C, H, W = processed_image.shape
        assert B == 1
        assert T == self.T
        assert C == 3

        speed = round(speed, 1)
        
        self.prompt_tp = prompt_tp
        self.prompt = prompt
        
        conversation_all = [
                {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                },
                {
                "role": "assistant",
                "content": [
                        {"type": "text", "text": "Waypoints:"},
                        ],
                },
        ]
        conv_batch_list = [conversation_all]
        questions = []
        for conv in conv_batch_list:
                for i in range(len(conv)):
                        questions.append(conv[i]['content'][0]['text'])
                        conv[i]['content'] = conv[i]['content'][0]['text']
                        
        cache_dir = f"pretrained/{(self.cfg.model.vision_model.variant.split('/')[1])}"
        # get absolute path from workspace dir
        cache_dir = to_absolute_path(cache_dir)
        model_path = f"{cache_dir}/conversation.py"
        if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=self.cfg.model.vision_model.variant, local_dir=cache_dir)
                
        # import conversation template from model_path
        spec = importlib.util.spec_from_file_location('get_conv_template', model_path)
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules['get_conv_template'] = conv_module
        spec.loader.exec_module(conv_module)
        
        if not hasattr(self, 'tmp_config'):
                self.tmp_config = AutoConfig.from_pretrained(self.cfg.model.vision_model.variant, trust_remote_code=True)
                image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
                patch_size = self.tmp_config.vision_config.patch_size
                
                self.num_image_token = int((image_size // patch_size) ** 2 * (self.tmp_config.downsample_ratio ** 2))
                
        prompt_batch_list = []
        for idx, conv in enumerate(conv_batch_list):
                question = questions[idx]
                if '<image>' not in question:
                        question = '<image>\n' + question
                template = conv_module.get_conv_template('internlm2-chat')
                template_inference = None
                
                template_inference = conv_module.get_conv_template('internlm2-chat')
                for conv_part_idx, conv_part in enumerate(conv):
                        if conv_part['role'] == 'assistant':
                                template.append_message(template.roles[1], None)
                        elif conv_part['role'] == 'user':
                                if conv_part_idx == 0 and '<image>' not in conv_part['content']:
                                        # add image token
                                        conv_part['content'] = '<image>\n' + conv_part['content']
                                template.append_message(template.roles[0], conv_part['content'])
                        else:
                                raise ValueError(f"Role {conv_part['role']} not supported")
                            
                query = template.get_prompt()
                # remove system prompt
                system_prompt = template.system_template.replace('{system_message}', template.system_message) + template.sep
                query = query.replace(system_prompt, '')
                
                IMG_START_TOKEN='<img>'
                IMG_END_TOKEN='</img>'
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
                num_patches_all = 2

                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_all + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
                prompt_batch_list.append(query)
                
        prompt_tokenized = self.tokenizer(prompt_batch_list, padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        prompt_tokenized_ids = prompt_tokenized["input_ids"]
        prompt_tokenized_char_offsets = prompt_tokenized["offset_mapping"].view(1, -1, 2)
        prompt_tokenized_valid = prompt_tokenized["input_ids"] != self.tokenizer.pad_token_id
        prompt_tokenized_mask = prompt_tokenized_valid
        
        ll = LanguageLabel(
                phrase_ids=prompt_tokenized_ids.to(self.device),
                phrase_valid=prompt_tokenized_valid.to(self.device),
                phrase_mask=prompt_tokenized_mask.to(self.device),
                placeholder_values=placeholder_batch_list,
                language_string=prompt_batch_list,
                loss_masking=None,
        )

        self.DrivingInput["camera_images"] = processed_image.to(self.device).bfloat16()
        self.DrivingInput["image_sizes"] = image_sizes
        self.DrivingInput["camera_intrinsics"] = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), 1, dim=0).view(1, 3, 3).float().to(self.device),
        self.DrivingInput["camera_extrinsics"] = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), 1, dim=0).view(1, 4, 4).float().to(self.device),
        self.DrivingInput["vehicle_speed"] = result['speed']
        self.DrivingInput["target_point"] = result['target_point'].to(self.device)
        self.DrivingInput["prompt"] = ll
        self.DrivingInput["prompt_inference"] = ll
        self.DrivingInput["driver_id"] = [self.driver_id]
    
        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            return control

        # Need to run this every step for GPS filtering
        tick_data = self.tick(input_data)

        
        # initialize DrivingInput with dict self.DrivingInput
        model_input = DrivingInput(**self.DrivingInput)
        pred_speed_wps, pred_route, language, dreaming_actions, dreaming_action_log_prob, dreaming_action_entropy = self.model.forward(model_input, return_language=False, rl_mode=True, deterministic=True)
        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None
        self.pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        self.pred_route = pred_route.float() if pred_route is not None else None

        # prepare velocity input
        gt_velocity = tick_data['speed']
        
        steer, throttle, brake, desired_speed = self.control_pid_personalization(pred_route, gt_velocity, pred_speed_wps, dreaming_actions[0])
        action = [steer, throttle, brake]

        if DEBUG and self.step % 5 == 0:
            tvec = None
            rvec = None

            if HD_VIZ:
                self.camera_for_viz = self.hd_cam_for_viz
                tvec = np.array([[0.0, 3.5, 5.5]], np.float32)

                cam_rots = [0.0, -15.0, 0.0]
                rot_matrix = get_rotation_matrix(-cam_rots[0], -cam_rots[1], cam_rots[2])
                rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()

            W=self.camera_for_viz.shape[1]
            H=self.camera_for_viz.shape[0]
            camera_intrinsics = np.asarray(get_camera_intrinsics(W,H,110))

            # bgr to rgb
            self.camera_for_viz = cv2.cvtColor(self.camera_for_viz, cv2.COLOR_BGR2RGB)

            # draw the predicted waypoints
            image = Image.fromarray(self.camera_for_viz)
            draw = ImageDraw.Draw(image)

            if self.target_points is not None:
                target_point_img_coords = project_points(self.target_points, camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in target_point_img_coords:
                    # in blue
                    draw.ellipse((points_2d[0]-4, points_2d[1]-4, points_2d[0]+4, points_2d[1]+4), fill=(0, 0, 255, 255))

            if pred_route is not None:
                pred_route_img_coords = project_points(pred_route[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_route_img_coords:
                        draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(255, 0, 0, 255))
            
            if pred_speed_wps is not None:
                pred_speed_wps_img_coords = project_points(pred_speed_wps[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_speed_wps_img_coords:
                        draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(0, 255, 0, 255))
            
            if language is not None:
                # write the language to the bottom of the image
                black_box = Image.new('RGBA', (W, 600), (0, 0, 0, 255))
                # concatenate the images
                image_all = Image.new('RGBA', (W, H+600))
                image_all.paste(image, (0, 0))
                image_all.paste(black_box, (0, H))
                image = image_all
                draw = ImageDraw.Draw(image)

                if HD_VIZ:
                    font_size = 50
                    line_width = 60
                    y_dist = 60
                    y_start = H + 20
                else:
                    font_size = 20
                    line_width = 100
                    y_dist = 30
                    y_start = H + 20
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", font_size)
                import textwrap
                lines = textwrap.wrap(f"Prompt: {self.prompt}", width=line_width)
                for idx, line in enumerate(lines):
                    draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

                y_start = y_start + y_dist*(idx+1)
                lines = textwrap.wrap(f"Chosen Action [steer throttle brake]: {action}", width=line_width)
                for idx, line in enumerate(lines):
                    draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

                y_start = y_start + y_dist*(idx+1)
                lines = textwrap.wrap(f"Chosen residual action: {dreaming_actions[0]}", width=line_width)
                for idx, line in enumerate(lines):
                    draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

                y_start = y_start + y_dist*(idx+1)
                lines = textwrap.wrap(f"Desired Speed: {desired_speed}", width=line_width)
                for idx, line in enumerate(lines):
                    draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

                y_start = y_start + y_dist*(idx+1)
                lines = textwrap.wrap(f"Driver ID: {self.driver_id}", width=line_width)
                for idx, line in enumerate(lines):
                    draw.text((10, y_start + y_dist*(idx)), line, font=font, fill=(255, 255, 255, 255))

            # save
            image.save(str(self.save_path_img / f"{self.step}.png"))
            
        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1
            print(f"force_move: {self.force_move}")

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control
            
        metric_info = self.get_metric_info(input_data, timestamp)
        self.metric_info[self.step] = metric_info
        if self.save_path_metric is not None and self.step % 1 == 0:
            # metric info
            outfile = open(str(self.save_path_metric / "metric_info.json"), 'w')
            json.dump(self.metric_info, outfile, indent=4)
            outfile.close()

        return control

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert route_waypoints.size(0) == 1
        route_waypoints = route_waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        # m / s required to drive
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        route_interp = self.interpolate_waypoints(route_waypoints[0:10])

        steer = self.turn_controller.step(route_interp, speed)

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake, desired_speed

    def control_pid_personalization(self, route_waypoints, velocity, speed_waypoints, dreaming_action):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert route_waypoints.size(0) == 1
        route_waypoints = route_waypoints[0].data.cpu().numpy()
        speed = velocity[0].data.cpu().numpy()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        # m / s required to drive
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0
        
        dreaming_delta_speed = np.clip(dreaming_action.to(torch.float32).cpu().numpy()[0][0], 0.5, 1.5)
        desired_speed = dreaming_delta_speed * desired_speed

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        route_interp = self.interpolate_waypoints(route_waypoints[0:10])
        steer = self.turn_controller.step(route_interp, speed)

        dreaming_delta_steer = np.clip(dreaming_action.to(torch.float32).cpu().numpy()[0][1], -0.3, 0.3)
        steer = steer + dreaming_delta_steer

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake, desired_speed

    # In: Waypoints NxD
    # Out: Waypoints NxD equally spaced 0.1 across D
    def interpolate_waypoints(self, waypoints):
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]

        dists = np.linalg.norm(waypoints-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4 # Prevents dists not being strictly increasing

        interp = PchipInterpolator(dists, waypoints, axis=0)

        x = np.arange(0.1, dists[-1], 0.1)

        interp_points = interp(x)

        # There is a possibility that all points are at 0, meaning there is no point distanced 0.1
        # In this case we output the last (assumed to be furthest) waypoint.
        if interp_points.shape[0] == 0:
                interp_points = waypoints[None, -1]

        return interp_points
    
    def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
        """
        Gets called after a route finished.
        The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
        Also writes logging files to disk.
        """

        del self.model
        del self.config

    def predict_other_actors_bounding_boxes(self, plant, actor_list, ego_vehicle_location, num_future_frames):
        """
            Predict the future bounding boxes of actors for a given number of frames.

            Args:
                plant (bool): Whether to use PlanT.
                actor_list (list): A list of actors (e.g., vehicles) in the simulation.
                ego_vehicle_location (carla.Location): The current location of the ego vehicle.
                num_future_frames (int): The number of future frames to predict.

            Returns:
                dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
            """
        predicted_bounding_boxes = {}

        if not plant:
        # Filter out nearby actors within the detection radius, excluding the ego vehicle
            nearby_actors = [
                actor for actor in actor_list if actor.id != self._vehicle.id and
                actor.get_location().distance(ego_vehicle_location) < self.config.detection_radius
            ]

            # If there are nearby actors, calculate their future bounding boxes
            if nearby_actors:
                # Get the previous control inputs (steering, throttle, brake) for the nearby actors
                previous_controls = [actor.get_control() for actor in nearby_actors]
                previous_actions = np.array([[control.steer, control.throttle, control.brake] for control in previous_controls])

                # Get the current velocities, locations, and headings of the nearby actors
                velocities = np.array([actor.get_velocity().length() for actor in nearby_actors])
                locations = np.array([[actor.get_location().x,
                                    actor.get_location().y,
                                    actor.get_location().z] for actor in nearby_actors])
                headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

                # Initialize arrays to store future locations, headings, and velocities
                future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
                future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
                future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

                # Forecast the future locations, headings, and velocities for the nearby actors
                for i in range(num_future_frames):
                    locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                        locations, headings, velocities, previous_actions)
                    future_locations[i] = locations.copy()
                    future_velocities[i] = velocities.copy()
                    future_headings[i] = headings.copy()

                # Convert future headings to degrees
                future_headings = np.rad2deg(future_headings)

                # Calculate the predicted bounding boxes for each nearby actor and future frame
                for actor_idx, actor in enumerate(nearby_actors):
                    predicted_actor_boxes = []

                    for i in range(num_future_frames):
                        # Calculate the future location of the actor
                        location = carla.Location(x=future_locations[i, actor_idx, 0].item(),
                                                y=future_locations[i, actor_idx, 1].item(),
                                                z=future_locations[i, actor_idx, 2].item())

                        # Calculate the future rotation of the actor
                        rotation = carla.Rotation(pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                        # Get the extent (dimensions) of the actor's bounding box
                        extent = actor.bounding_box.extent
                        # Otherwise we would increase the extent of the bounding box of the vehicle
                        extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                        # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                        # uncertainty during forecasting
                        s = self.config.high_speed_min_extent_x_other_vehicle
                        extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                s,
                                self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                        extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                            i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                                self.config.high_speed_min_extent_y_other_vehicle,
                                self.config.high_speed_extent_y_factor_other_vehicle * float(i) / float(num_future_frames))

                        # Create the bounding box for the future frame
                        bounding_box = carla.BoundingBox(location, extent)
                        bounding_box.rotation = rotation

                        # Append the bounding box to the list of predicted bounding boxes for this actor
                        predicted_actor_boxes.append(bounding_box)

                    # Store the predicted bounding boxes for this actor in the dictionary
                    predicted_bounding_boxes[actor.id] = predicted_actor_boxes

        return predicted_bounding_boxes
    

    def forecast_ego_vehicle_from_predicted_waypoints(self, current_ego_transform, current_ego_speed, route_waypoints, speed_waypoints):
        """
        Forecast future ego states using predicted route and speed waypoints.

        Args:
            current_ego_transform (carla.Transform): Current transform of the ego vehicle.
            current_ego_speed (float): Current speed (m/s).
            route_waypoints (torch.Tensor): Shape [1, N, 2], predicted route positions.
            speed_waypoints (torch.Tensor): Shape [1, M, 2], predicted waypoints from speed predictor.

        Returns:
            list: List of carla.BoundingBox representing predicted ego vehicle poses.
        """
        assert route_waypoints.shape[0] == 1 and speed_waypoints.shape[0] == 1

        route_waypoints = route_waypoints[0].cpu().numpy()
        speed_waypoints = speed_waypoints[0].cpu().numpy()

        location = np.array([
            current_ego_transform.location.x,
            current_ego_transform.location.y,
            current_ego_transform.location.z
        ])
        heading_angle = np.array([np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([current_ego_speed.cpu().numpy()])

        future_bounding_boxes = []

        for t in range(min(len(route_waypoints), len(speed_waypoints) - 2)):
            # Estimate desired speed using speed waypoints
            desired_speed = np.linalg.norm(speed_waypoints[t] - speed_waypoints[t + 2]) * 2.0

            # Compute control actions
            brake = (desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio)
            delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
            throttle = self.speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
            throttle = throttle if not brake else 0.0

            route_interp = self.interpolate_waypoints(route_waypoints[t:t+5])  # 5-waypoint local lookahead
            steer = self.turn_controller.step(route_interp, speed)
            steer = np.clip(steer, -1.0, 1.0)

            # Action format: [steer, throttle, brake]
            action = np.array([steer, throttle, float(brake)])

            # Predict next state
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(location, heading_angle, speed, action)

            # Compute bounding box
            heading_angle_degrees = np.rad2deg(heading_angle).item()
            extent = self._vehicle.bounding_box.extent
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

            transform = carla.Transform(
                carla.Location(x=location[0].item(), y=location[1].item(), z=location[2].item())
            )
            bbox = carla.BoundingBox(transform.location, extent)
            bbox.rotation = carla.Rotation(pitch=0, yaw=heading_angle_degrees, roll=0)

            future_bounding_boxes.append(bbox)

        return future_bounding_boxes
    
    def get_metric_info(self, input_data=None, timestamp=None):
        output = super().get_metric_info()
        other_vehicles_info = []
        world = self.hero_actor.get_world()

        # Iterate over all vehicle actors in the scene
        for actor in world.get_actors().filter('vehicle.*'):
            # Skip the ego vehicle
            if actor.id == self.hero_actor.id:
                continue

            # Extract vehicle information
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            speed_kmh = velocity.length()

            # Store bounding box info and vehicle state
            vehicle_data = {
                "id": actor.id,
                "type": actor.type_id,
                "speed_kmh": speed_kmh,
                "transform": {
                    "location": {
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z,
                    },
                    "rotation": {
                        "roll": transform.rotation.roll,
                        "pitch": transform.rotation.pitch,
                        "yaw": transform.rotation.yaw,
                    }
                },
                "velocity": {
                    "x": velocity.x,
                    "y": velocity.y,
                    "z": velocity.z,
                },
                "bounding_box_extent": {
                    "x": actor.bounding_box.extent.x,
                    "y": actor.bounding_box.extent.y,
                    "z": actor.bounding_box.extent.z,
                }
            }
            other_vehicles_info.append(vehicle_data)

        # Add all other vehicles' info to the output
        output['other_vehicles'] = other_vehicles_info

        return output

# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model.
    # Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    '''
        For now we use the same internal state as the measurement state
        :param vehicle_state: VehicleState vehicle state variable containing
                                                    an internal state of the vehicle from the filter
        :return: np array: describes the vehicle state as numpy array.
                                             0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
        '''
    return vehicle_state


def state_mean(state, wm):
    '''
        We use the arctan of the average of sin and cos of the angle to calculate
        the average of orientations.
        :param state: array of states to be averaged. First index is the timestep.
        :param wm:
        :return:
        '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    '''
    We use the arctan of the average of sin and cos of the angle to
    calculate the average of orientations.
    :param state: array of states to be averaged. First index is the
    timestep.
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
