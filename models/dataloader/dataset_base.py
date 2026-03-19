"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""
import datetime
import glob
import gzip
import os
import pickle as pkl
import random
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import ujson
from imgaug import augmenters as ia
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from tqdm import tqdm
import carla

import models.utils.transfuser_utils as t_u
from models.utils.custom_types import DatasetOutput
from models.utils.projection import get_camera_intrinsics, project_points
from models.utils.kinematic_bicycle_model import KinematicBicycleModel
from team_code.config import GlobalConfig
from team_code.lateral_controller import LateralPIDController
from team_code.longitudinal_controller import LongitudinalLinearRegressionController

VIZ_DATA = False

class BaseDataset(Dataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Base class for the dataset.
    """

    def __init__(self,
            evaluation = False,
            **cfg,
        ):
        for key, value in cfg.items():
            setattr(self, key, value)

        self.tfs = image_augmenter(prob=self.img_augmentation_prob)

        self.carla_frame_rate = 20
        self.dataset_frame_rate = 4
        self.num_future_frames_carla_fps = int(self.carla_frame_rate * 2.5)

        self.config = GlobalConfig()
        self.bicycle_model = KinematicBicycleModel(self.carla_frame_rate)
        self._turn_controller = LateralPIDController(self.config)
        self._longitudinal_controller = LongitudinalLinearRegressionController(self.config)

        filter_infractions_per_route = True

        self.rgb_folder = 'rgb'
        
        self.images = []
        self.boxes = []
        self.measurements = []
        self.sample_start = []
        self.augment_exists = []
        self.alternative_trajectories = []

        self.temporal_measurements = []

        total_routes = 0
        perfect_routes = 0
        crashed_routes = 0

        fail_reasons = {}

        repo_path = os.environ.get('WORK_DIR', '')
        self.repo_path = repo_path

        augment_exist = False

        route_dirs = glob.glob(f"{repo_path}/" + self.data_path + '/data/dmw/*/*/*/')
        print(f'Found {len(route_dirs)} routes in {repo_path}/{self.data_path}')

        split_percentage = 0.8
        if self.split == "train":
            route_dirs = route_dirs[:int(split_percentage * len(route_dirs))]
        elif self.split == "val":
            route_dirs = route_dirs[:int(split_percentage * len(route_dirs))]
        
        print(f'Use {len(route_dirs)} routes.')
        scenario_count = {}
        
        for sub_root in tqdm(route_dirs, file=sys.stdout):

            route_dir = sub_root

            if filter_infractions_per_route:
                if not os.path.isfile(route_dir + '/results.json.gz'):
                    total_routes += 1
                    crashed_routes += 1
                    if "no_results.json" not in fail_reasons:
                        fail_reasons["no_results.json"] = 1
                    else:
                        fail_reasons["no_results.json"] += 1
                    continue

                with gzip.open(route_dir + '/results.json.gz', 'rt') as f:
                    total_routes += 1
                    try:
                        results_route = ujson.load(f)
                    except Exception as e:
                        print(f"Error in {route_dir}")
                        print(e)
                        if "results.json_load_error" not in fail_reasons:
                            fail_reasons["results.json_load_error"] = 1
                        else:
                            fail_reasons["results.json_load_error"] += 1
                        continue

                if results_route['scores']['score_composed'] < 100.0:  # we also count imperfect runs as failed (except minspeedinfractions)
                    cond1 = results_route['scores']['score_route'] > 94.0  # we allow 6% of the route score to be missing
                    cond2 = results_route['num_infractions'] == (len(results_route['infractions']['min_speed_infractions']) + len(results_route['infractions']['outside_route_lanes']))
                    if not (cond1 and cond2):  # if the only problem is minspeedinfractions, keep it
                        crashed_routes += 1
                        if "route_crashed" not in fail_reasons:
                            fail_reasons["route_crashed"] = 1
                        else:
                            fail_reasons["route_crashed"] += 1
                        continue

            perfect_routes += 1

            num_seq = len(os.listdir(route_dir + f'/{self.rgb_folder}'))

            route_config_path = route_dir.split(self.data_path)[-1].lstrip('/')
            route_id = route_config_path.split("Rep0_")[1].split("_")[0]
            path_parts = route_config_path.split('/')[:-1]  # This is a list
            relative_dir = '/'.join(path_parts)  # Convert list back to path string
            route_config_path = os.path.join(self.repo_path, relative_dir, f"{route_id}.xml")
            scenario_type = extract_scenario_type(route_config_path)[0]['type']
            if scenario_type not in scenario_count:
                scenario_count[scenario_type] = 0
            scenario_count[scenario_type] += 1
            if scenario_count[scenario_type] > 1:
                continue

            for seq in range(self.skip_first_n_frames, num_seq - self.pred_len - self.hist_len - 1):
                image = []
                box = []
                measurement = []

                measurement_file = route_dir + '/measurements' + f'/{(seq + self.hist_len-1):04}.json.gz'

                if evaluation and measurement_file not in self.all_eval_samples:
                    continue
                
                if self.bucket_name is not None and self.bucket_name != "all":
                    measurement_file_path = Path(measurement_file)
                    if str(measurement_file_path.parent) in run_id_dict:
                        if measurement_file_path.name not in run_id_dict[str(measurement_file_path.parent)]:
                            if "measurement_file_not_in_bucket" not in fail_reasons:
                                fail_reasons["measurement_file_not_in_bucket"] = 1
                            else:
                                fail_reasons["measurement_file_not_in_bucket"] += 1
                            continue
                    else:
                        if "measurement_folder_not_in_bucket" not in fail_reasons:
                            fail_reasons["measurement_folder_not_in_bucket"] = 1
                        else:
                            fail_reasons["measurement_folder_not_in_bucket"] += 1
                        continue

                skip = False
                for idx in range(self.hist_len):
                    image.append(route_dir +  f'/{self.rgb_folder}' + (f'/{(seq + idx):04}.jpg'))

                if skip:
                    if "file_not_found" not in fail_reasons:
                        fail_reasons["file_not_found"] = 1
                    else:
                        fail_reasons["file_not_found"] += 1
                    continue

                measurement.append(route_dir + '/measurements')
                box.append(route_dir + '/boxes')

                self.images.append(image)
                self.boxes.append(box)
                self.measurements.append(measurement)
                self.sample_start.append(seq)
                self.augment_exists.append(augment_exist)

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)

        self.sample_start = np.array(self.sample_start)
        print(f'[{self.split} samples]: Loading {len(self.images)} images from {self.data_path} for bucket {self.bucket_name}')
        print('Total amount of routes:', total_routes)
        print('Crashed routes:', crashed_routes)
        print('Perfect routes:', perfect_routes)
        print('Fail reasons:', fail_reasons)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.images.shape[0]
    

    def load_current_and_future_measurements(self, measurements, sample_start):
        loaded_measurements = []

        # Since we load measurements for future time steps, we load and store them separately
        for i in range(self.hist_len):
            measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(measurement_file, 'rt') as f1:
                measurements_i = ujson.load(f1)
            loaded_measurements.append(measurements_i)

        end = self.pred_len + self.hist_len
        start = self.hist_len

        for i in range(start, end):
            try:
                measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

                with gzip.open(measurement_file, 'rt') as f1:
                    measurements_i = ujson.load(f1)
                loaded_measurements.append(measurements_i)
            except FileNotFoundError:
                # If the file is not found, we just use the last available measurement
                print(f"File not found: {measurement_file}")
                loaded_measurements.append(loaded_measurements[-1])
        current_measurement = loaded_measurements[self.hist_len - 1]
        measurement_file_current = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + start-1):04}.json.gz')
        return loaded_measurements, current_measurement, measurement_file_current

    def load_waypoints(self, data, loaded_measurements, aug_translation=0.0, aug_rotation=0.0):

        waypoints = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=aug_translation,
                                                                        yaw_augmentation=aug_rotation)
        data['waypoints'] = np.array(waypoints[1:-1])

        waypoints_org = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=0,
                                                                        yaw_augmentation=0)
        data['waypoints_org'] = np.array(waypoints_org[1:-1])

        # 1D waypoints: only consider distance between waypoints
        waypoints_1d = [np.linalg.norm(waypoints_org[i+1] - waypoints_org[i]) for i in range(len(waypoints_org)-1)]
        # cumsum to get the distance from the start
        waypoints_1d = np.cumsum(waypoints_1d)
        waypoints_1d = [[x, 0] for x in waypoints_1d]
        data['waypoints_1d'] = np.array(waypoints_1d[:-1]).reshape(-1, 2)

        waypoints = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints]
        data['ego_waypoints'] = np.array(waypoints[:-1])
        
        waypoints_org = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints_org]
        data['ego_waypoints_org'] = np.array(waypoints_org[:-1])

        return data
    
    def load_route(self, data, current_measurement, aug_translation=0.0, aug_rotation=0.0):
        route = current_measurement['route_original']
        route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        route_adjusted = np.array(current_measurement['route'])
        route_adjusted_org = self.augment_route(route_adjusted, y_augmentation=0, yaw_augmentation=0)
        route_adjusted = self.augment_route(route_adjusted, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        if len(route) < self.num_route_points:
            num_missing = self.num_route_points - len(route)
            route = np.array(route)
            # Fill the empty spots by repeating the last point.
            route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
        else:
            route = np.array(route[:self.num_route_points])
            
        route_adjusted = self.equal_spacing_route(route_adjusted)
        route_adjusted_org = self.equal_spacing_route(route_adjusted_org)
        route = self.equal_spacing_route(route)
        
        data['route'] = route
        data['route_adjusted_org'] = route_adjusted_org
        data['route_adjusted'] = route_adjusted

        return data
    
    def load_images(self, data, images, augment_sample=False):
        loaded_images = []
        loaded_images_org_size = []
        for i in range(self.hist_len):
            images_i = None
            images_path = str(images[i], encoding='utf-8')
            if augment_sample:
                images_path = images_path.replace('rgb', 'rgb_augmented')

            if not os.path.isfile(images_path):
                print(f"File not found: {images_path}")
                raise FileNotFoundError

            images_i = cv2.imread(images_path, cv2.IMREAD_COLOR)
            images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)

            if self.img_augmentation:
                images_i = self.tfs(image=images_i)
            
            image_org = images_i.copy()
            if self.cut_bottom_quarter or self.img_shift_augmentation:
                images_i = images_i[:int(images_i.shape[0] - (images_i.shape[0] * 4.8) // 16), :, :]

            loaded_images.append(images_i)
            loaded_images_org_size.append(image_org)
        
        processed_image = np.asarray(loaded_images)
        processed_image_org_size = np.asarray(loaded_images_org_size)

        # we want [T, N, C, H, W], T is the number of temporal frames, N is the number of cam views, C is the number of channels, H is the height and W is the width
        processed_image = np.transpose(processed_image, (0, 3, 1, 2)) # (T, C, H, W)
        processed_image_org_size = np.transpose(processed_image_org_size, (0, 3, 1, 2)) # (T, C, H, W)

        data['rgb'] = processed_image
        data['rgb_org_size'] = processed_image_org_size

        return data

    def load_current_and_future_boxes(self, boxes, sample_start):
        loaded_boxes = []

        for i in range(self.hist_len):
            box_file = str(boxes[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(box_file, 'rt') as f1:
                box_i = ujson.load(f1)
            loaded_boxes.append(box_i)

        for i in range(self.hist_len, self.pred_len + self.hist_len):
            try:
                box_file = str(boxes[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

                with gzip.open(box_file, 'rt') as f1:
                    box_i = ujson.load(f1)
                loaded_boxes.append(box_i)
            except FileNotFoundError:
                # If the file is not found, we just use the last available measurement
                print(f"File not found: {box_file}")
                loaded_boxes.append(loaded_boxes[-1])

        current_boxes = loaded_boxes[self.hist_len - 1]
        return loaded_boxes, current_boxes
    
    def get_navigational_conditioning(self, data, current_measurement, target_point, next_target_point):
        placeholder_values = {}
        target_options = []
                
        tp = [target_point, next_target_point]
        tp = np.array(tp)
        data['map_route'] = tp
        data['target_points'] = tp
        target_point1_round = np.round(data['target_points'][0], 2).tolist()
        target_point2_round = np.round(data['target_points'][1], 2).tolist()

        if 'target_point' in self.route_as:
            if 'target_point_language' in self.route_as:
                target_options.append(f"Target waypoint: 1:{target_point1_round} 2:{target_point2_round}")
            else:
                target_options.append(f"Target waypoint: <TARGET_POINT><TARGET_POINT>.")
                placeholder_values = {'<TARGET_POINT>': data['target_points']}
        if 'command' in self.route_as:
            # get distance from target_point
            dist_to_command = np.linalg.norm(target_point)
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
            command = map_command[current_measurement["command"]]
            next_command = map_command[current_measurement["next_command"]]
            if command != next_command:
                next_command = f' then {next_command}'
            else:
                next_command = ''
            if current_measurement["command"] == 4:
                command_str = f'Command: {command}{next_command}.'
            else:
                command_str = f'Command: {command} in {dist_to_command} meter{next_command}.'
            target_options.append(command_str)
            
            if self.use_lmdrive_commands:
                lmdrive_index = random.choice(command_template_mappings[current_measurement["command"]])
                lmdrive_command = random.choice(self.command_templates[str(lmdrive_index)])
                lmdrive_command = lmdrive_command.replace('[x]', str(dist_to_command))
                lm_command = f'Command: {lmdrive_command}.'
                target_options.append(lm_command)
        
        return target_options, placeholder_values

    def equal_spacing_route(self, points):
        route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
        shift = np.roll(route, 1, axis=0) # Shift by 1
        shift[0] = shift[1] # Set wraparound value to 0

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

        x = np.arange(0, 20, 1)
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points
    
    def visualise_cameras(
        self,
        batch: DatasetOutput,
        language, route, waypoints,
        options,
        name: str = "img",
        prompt=None,
        answer=None,
    ) -> np.ndarray:
        
        fov = 110

        img_front_np = batch.image_ff_org_size
        img_front_np = img_front_np.transpose(0, 2, 3, 1)

        all_images = [Image.fromarray((img_front_np[i])) for i in range(1)]

        all_draws = [ImageDraw.Draw(image) for image in all_images]
        
        # black image to be concatenated to the bottom of the image
        img_width = all_images[0].size[0]
        text_box = [Image.new("RGB", (img_width, 200), "black") for _ in range(1)]
        text_draw = [ImageDraw.Draw(image) for image in text_box]
        
        W=all_images[0].size[0]
        H=all_images[0].size[1]
        camera_intrinsics = np.asarray(get_camera_intrinsics(W,H,fov))

        for i in range(1):
            gt_waypoints_img_coords = project_points(batch.waypoints, camera_intrinsics)
            for points_2d in gt_waypoints_img_coords:
                all_draws[i].ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 255, 0, 255))

            if route is not None:
                pred_route_img_coords = project_points(route, camera_intrinsics)
                for points_2d in pred_route_img_coords:
                    all_draws[i].ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(255, 0, 0, 255))

            if language is not None:
                y_curr = 10
                text_draw[i].text((10, y_curr), f"Commentary: {language}", fill=(255, 255, 255, 255))
            if prompt is not None:
                text_draw[i].text((10, 30), f"Prompt: {prompt}", fill=(255, 255, 255, 255))
            if answer is not None:
                text_draw[i].text((10, 50), f"Answer: {answer}", fill=(255, 255, 255, 255))
                
        # concat text box to the bottom of the image
        
        if options is not None:
            all_all_images = [None for _ in range(len(options))]
            all_blacks = [None for _ in range(len(options))]
            for i, option in enumerate(options):
                wp_altern = option['waypoints']
                route_altern = option['route']
                if isinstance(route_altern, str) and route_altern == 'org':
                    route_altern = route
                img = all_images[0].copy()
                draw = ImageDraw.Draw(img)
                img_black = text_box[0].copy()
                draw_black = ImageDraw.Draw(img_black)
                gt_waypoints_img_coords = project_points(wp_altern, camera_intrinsics)
                for points_2d in gt_waypoints_img_coords:
                    draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 55, 0, 255))
                if route_altern is not None:
                    pred_route_img_coords = project_points(route_altern, camera_intrinsics)
                    for points_2d in pred_route_img_coords:
                        draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(55, 0, 0, 255))
                if language is not None:
                    draw_black.text((10, 80), f"Alternative Traj: {language}", fill=(255, 255, 255, 255))
                    draw_black.text((10, 100), f"Alternative Traj: {answer}", fill=(255, 255, 255, 255))
                    
                all_all_images[i] = img
                all_blacks[i] = img_black
            
        all_images = [Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_images, text_box)]
        if options is not None:
            all_images.extend([Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_all_images, all_blacks)])
        
        # concat all images
        viz_image_np = np.concatenate([np.array(image) for image in all_images], axis=0)
        viz_image = Image.fromarray(viz_image_np)
        
        current_dir = os.getcwd()
        
        Path("viz_images").mkdir(parents=True, exist_ok=True)
        # save the image
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_image.save(f"viz_images/{name}_{time}.png")

        return viz_image_np
    

    def get_indices_speed_angle(self, target_speed, brake):
        target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20, 24]  # v4 target speeds (0.72*speed limits) plus extra classes for obstacle scenarios and intersecions

        target_speed_bins = [x+0.001 for x in target_speeds[1:]]  
        target_speed_index = np.digitize(x=target_speed, bins=target_speed_bins)

        # Define the first index to be the brake action
        if brake:
            target_speed_index = 0
        else:
            target_speed_index += 1

        return target_speed_index

    def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                    np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0, y_augmentation]])
        route_aug = (rotation_matrix.T @ (route - translation).T).T
        return route_aug

    def augment_target_point(self, target_point, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        pos = np.expand_dims(target_point, axis=1)
        target_point_aug = rotation_matrix.T @ (pos - translation)
        return np.squeeze(target_point_aug)

    def parse_bounding_boxes(self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0):

        bboxes = []
        for current_box in boxes:
            # Ego car is always at the origin. We don't predict it.
            if current_box['class'] == 'ego_car':
                continue

            if 'position' not in current_box or 'extent' not in current_box:
                continue

            bbox, height = self.get_bbox_label(current_box, y_augmentation, yaw_augmentation)

            if current_box['class'] == 'traffic_light':
                # Only use/detect boxes that are red and affect the ego vehicle
                if not current_box['affects_ego']:
                    continue

            if current_box['class'] == 'stop_sign':
                # Don't detect cleared stop signs.
                if not current_box['affects_ego']:
                    continue

            bboxes.append(bbox)
        return bboxes

    def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0):
        # augmentation
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        position = np.array([[bbox_dict['position'][0]], [bbox_dict['position'][1]]])
        translation = np.array([[0.0], [y_augmentation]])

        position_aug = rotation_matrix.T @ (position - translation)

        x, y = position_aug[:2, 0]
        # center_x, center_y, w, h, yaw
        bbox = np.array([x, y, bbox_dict['extent'][0], bbox_dict['extent'][1], 0, 0, 0, 0, 0])
        bbox[4] = t_u.normalize_angle(bbox_dict['yaw'] - aug_yaw_rad)

        if bbox_dict['class'] == 'car':
            bbox[5] = bbox_dict['speed']
            bbox[6] = bbox_dict['brake']
            bbox[7] = 0
        elif bbox_dict['class'] == 'walker':
            bbox[5] = bbox_dict['speed']
            bbox[7] = 1
        elif bbox_dict['class'] == 'traffic_light':
            bbox[7] = 2
            if bbox_dict['state'] == 'Green':
                bbox[8] = 0
            elif bbox_dict['state'] == 'Red' or bbox_dict['state'] == 'Yellow':
                bbox[8] = 1
            else:
                bbox[8] = 2
        elif bbox_dict['class'] == 'stop_sign':
            bbox[7] = 3

        else:
            bbox = np.zeros(9)
        return bbox, bbox_dict['position'][2]

    def get_route_image(self, route, target_point):
        route_img = np.zeros((64, 64, 3), dtype=np.uint8)
        route_new = np.array(route, dtype=np.float32)
        route_new[:, 0] = -route_new[:, 0]*2 + 63
        route_new[:, 1] = route_new[:, 1]*2 + 32
        route_new = route_new.clip(0, 63)
        route_new = route_new.astype(np.int32)
        route_img[route_new[:, 0], route_new[:, 1], :] = 255

        return route_img

    def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """transform waypoints to be origin at ego_matrix"""
        origin = measurements[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for index in range(len(measurements)):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])

        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad), np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug

    def get_bbs(self, actors, future_nearby_actors_by_id, future_nearby_actors_used_time_stamps_by_id, all_ego_positions, all_ego_yaws):
        if not actors:
            return  {}
        
        all_bbs_by_id = {}
        for actor in actors:
                future_actor = future_nearby_actors_by_id[actor['id']]
                future_nearby_actors_used_time_stamps = [0] + [t+1 for t in future_nearby_actors_used_time_stamps_by_id[actor['id']]]
                actor_all_times = [actor] + future_actor
                all_positions = np.array([actor['position'] for actor in actor_all_times])
                all_yaws = np.array([actor['yaw'] for actor in actor_all_times])
                # convert all positions (current and future) (actor_all_times[i]['position']) first to global and then back to local of the current ego frame
                all_positions_global = np.array([t_u.conversion_2d(pos[:2], all_ego_positions[future_nearby_actors_used_time_stamps[t]], -all_ego_yaws[future_nearby_actors_used_time_stamps[t]]) for t, pos in enumerate(all_positions)])
                all_positions_local = np.array([t_u.inverse_conversion_2d(pos, all_ego_positions[0], all_ego_yaws[0]) for pos in all_positions_global])

                all_yaws_global = np.array([yaw + all_ego_yaws[t] for t, yaw in enumerate(all_yaws)])
                all_yaws_local = np.array([t_u.normalize_angle(yaw - all_ego_yaws[0]) for yaw in all_yaws_global])

                # get the bounding box for each actor for each timestep
                all_bbs = []
                extent_add_safety = 0
                if actor['class'] == 'walker':
                    extent_add_safety = 0.5
                
                for t in range(len(actor_all_times)):
                    location = carla.Location(
                        x=all_positions_local[t][0],
                        y=all_positions_local[t][1],
                        z=all_positions[t][2]
                    )
                    rotation = carla.Rotation(
                        pitch=0,
                        yaw=np.rad2deg(all_yaws_local[t]),
                        roll=0
                    )
                    extent = carla.Vector3D(x=actor_all_times[t]['extent'][0]+extent_add_safety, y=actor_all_times[t]['extent'][1]+extent_add_safety, z=actor_all_times[t]['extent'][2])
                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation
                    all_bbs.append(bounding_box)
                all_bbs_by_id[actor['id']] = all_bbs

        return all_bbs_by_id    

    def _dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
        
    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)
        
    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self._dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (
            abs(self._dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
            abs(self._dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
            abs(self._dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
        )

        obb2_projection = (
            abs(self._dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
            abs(self._dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
            abs(self._dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
        )

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection
        
    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (self.get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):
            
            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or 
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(),obb2.rotation.get_up_vector()), obb1, obb2)):
            
            return False

        # If no separating plane is found, the OBBs intersect
        return True
    

    def forecast_vehicles(self, ego_actor, next_ego_actor_by_id, ego_position, ego_yaw, route=None, speeds_to_follow=None, desired_throttle=None, brake_probability=None, target_speed=None, return_final_speed=False, return_gt_speeds=False, use_wps_speed_controller=False):
        
        predicted_bounding_boxes = {}
        self._turn_controller.save_state()

        desired_speeds = []

        # between speeds_to_follow and target_speed and desired_throttle only one can be set
        if speeds_to_follow is not None:
            assert desired_throttle is None
            assert target_speed is None
        if desired_throttle is not None:
            assert speeds_to_follow is None
            assert target_speed is None
        if target_speed is not None:
            assert speeds_to_follow is None
            assert desired_throttle is None

        next_ego_actor = next_ego_actor_by_id[ego_actor["id"]]
        current_actions = np.array([[ego_actor["steer"], ego_actor["throttle"], ego_actor["brake"]]])
        actions_future = np.array([[actor["steer"], actor["throttle"], actor["brake"]] for actor in next_ego_actor])
        all_actions = np.concatenate([current_actions, actions_future])
        all_locations_global = np.concatenate([np.array([np.array(ego_actor["matrix"])[:3, 3]]), np.array([np.array(actor["matrix"])[:3, 3] for actor in next_ego_actor])])
        all_locations_local = np.array([t_u.inverse_conversion_2d(pos[:2], ego_position, ego_yaw) for pos in all_locations_global])
        all_yaws = np.concatenate([np.array([ego_actor["yaw"]]), np.array([actor["yaw"] for actor in next_ego_actor])])
        all_speeds = np.concatenate([np.array([ego_actor["speed"]]), np.array([actor["speed"] for actor in next_ego_actor])])

        # linearly interpolate to have 20 fps instead of 5 fps
        ratio = self.dataset_frame_rate / self.carla_frame_rate # we need ad datapoint every 0.2 seconds
        interp_steers = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 0])
        interp_throttles = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 1])
        interp_brakes = np.interp(np.arange(0, len(all_actions), ratio), np.arange(0, len(all_actions)), all_actions[:, 2])
        interp_location_global_x = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 0])
        interp_location_global_y = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 1])
        interp_location_global_z = np.interp(np.arange(0, len(all_locations_global), ratio), np.arange(0, len(all_locations_global)), all_locations_global[:, 2])
        interp_location_global = np.array([interp_location_global_x, interp_location_global_y, interp_location_global_z]).T
        interp_yaws = np.interp(np.arange(0, len(all_yaws), ratio), np.arange(0, len(all_yaws)), all_yaws)
        interp_speeds = np.interp(np.arange(0, len(all_speeds), ratio), np.arange(0, len(all_speeds)), all_speeds)

        # Get the previous control inputs (steering, throttle, brake) for the nearby actors
        previous_actions = np.array([ego_actor["steer"], ego_actor["throttle"], ego_actor["brake"]])

        # Get the current velocities, locations, and headings of the nearby actors
        velocities = np.array([ego_actor["speed"]])
        locations = np.array(np.asarray(ego_actor["matrix"])[:3, 3])
        headings = np.array([ego_actor["yaw"]+ego_yaw])

        # Initialize arrays to store future locations, headings, and velocities
        future_locations = np.zeros((self.num_future_frames_carla_fps, 3), dtype='float')
        future_headings = np.zeros((self.num_future_frames_carla_fps), dtype='float')
        future_velocities = np.zeros((self.num_future_frames_carla_fps), dtype='float')

        # Forecast the future locations, headings, and velocities for the nearby actors
        for i in range(self.num_future_frames_carla_fps):

            locations, headings, velocities = self.bicycle_model.forecast_ego_vehicle(locations, headings, velocities, previous_actions)
            previous_actions = np.array([interp_steers[i], interp_throttles[i], interp_brakes[i]])

            future_locations[i] = locations.copy()
            future_velocities[i] = velocities.copy()
            future_headings[i] = headings.copy()

            if route is not None:
                location_global = locations
                speed = velocities

                # the +1 is important, as otherwise the point is too close and the controller will not work -> leads to oscillations
                closest_route_point = np.argmin(np.linalg.norm(location_global[:2] - route[:, :2], axis=1)) + 1
                if closest_route_point >= len(route):
                    route = None # use original controll instead
                else:
                    closest_route_point = min(closest_route_point, len(route)-1)
                    route_ahead = route[closest_route_point:]

                    # get the steering angle from the PID and overwrite the original steering angle
                    steering = self._turn_controller.step(route_ahead, speed, locations[:2], headings, inference_mode=True)
                    previous_actions[0] = steering
            
            if use_wps_speed_controller:
                wps_global = interp_location_global[i:]
                wps = np.array([t_u.inverse_conversion_2d(wp[:2], locations[:2], headings) for wp in wps_global])[::(self.carla_frame_rate//self.dataset_frame_rate)]
                one_second = self.dataset_frame_rate
                half_second = one_second // 2
                one_half_second = one_second + half_second

                if len(wps) >= one_second:
                    desired_speed = np.linalg.norm(wps[half_second - 2] - wps[one_second - 2] - wps[0]) * 2.0
                    desired_speeds.append(desired_speed)

                    throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, desired_speed, velocities[0])

                    previous_actions[1] = throttle
                    previous_actions[2] = control_brake

            if speeds_to_follow is not None:
                desired_speed = speeds_to_follow[i]
                throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, desired_speed, velocities[0])
                previous_actions[1] = throttle
                previous_actions[2] = control_brake
            
            if desired_throttle is not None:
                previous_actions[1] = desired_throttle
                previous_actions[2] = 0

            if brake_probability is not None:
                if random.random() < brake_probability:
                    previous_actions[1] = 0
                    previous_actions[2] = 1
                else:
                    previous_actions[1] = 0
                    previous_actions[2] = 0

            if target_speed is not None:
                throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(False, target_speed, velocities[0])
                previous_actions[1] = throttle
                previous_actions[2] = control_brake
                
        # Convert future headings to degrees
        future_headings = np.rad2deg(future_headings)

        # Convert global coordinates to egocentric coordinates
        ego_position = np.array(ego_position)
        ego_orientation = np.array(ego_yaw)
        for time_step in range(future_locations.shape[0]):
            target_point_2d = future_locations[time_step, :2]

            ego_target_point = t_u.inverse_conversion_2d(target_point_2d, ego_position, ego_orientation).tolist()
            future_locations[time_step, :2] = ego_target_point
            future_headings[time_step] -= np.rad2deg(ego_yaw)


            # Calculate the predicted bounding boxes for each future frame
            predicted_actor_boxes = []
            for i in range(self.num_future_frames_carla_fps):         
                location = carla.Location(
                    x=future_locations[i, 0].item(),
                    y=future_locations[i, 1].item(),
                    z=future_locations[i, 2].item()
                )
                rotation = carla.Rotation(
                    pitch=0,
                    yaw=future_headings[i],
                    roll=0
                )
                extent = ego_actor["extent"]
                extent = carla.Vector3D(x=extent[0], y=extent[1], z=extent[2])

                # Create the bounding box for the future frame
                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation

                # Append the bounding box to the list of predicted bounding boxes for this actor
                predicted_actor_boxes.append(bounding_box)

            # Store the predicted bounding boxes for this actor in the dictionary
            # sample the bounding boxes every 4 frames
            predicted_actor_boxes = predicted_actor_boxes[::(self.carla_frame_rate//self.dataset_frame_rate)]

            predicted_bounding_boxes[ego_actor["id"]] = predicted_actor_boxes
        self._turn_controller.load_state()

        return_tuple = (predicted_bounding_boxes,)
        
        if return_final_speed:
            final_speed = future_velocities[-1]
            final_speed = round(final_speed, 1)
            return_tuple += (final_speed,)

        if return_gt_speeds:
            return_tuple += (future_velocities,)
        if len(return_tuple) == 1:
            return_tuple = predicted_bounding_boxes
        return return_tuple
        
def image_augmenter(prob=0.2, cutout=False):
    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
        ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
    ]

    if cutout:
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

    augmenter = ia.Sequential(augmentations, random_order=True)

    return augmenter


def get_camera_intrinsics(w, h, fov):
    """
    Get camera intrinsics matrix from width, height and fov.
    Returns:
        K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
            the carla camera.
    """

    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    K = torch.tensor(K, dtype=torch.float32)
    return K

def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinic calibration matrix for
            the carla camera. The extriniscs are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """

    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]

    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics

def get_camera_distortion():
    """
    Get camera distortion matrix for the carla camera.
    distortion: A float32 tensor of shape ``[14 + 1]`` containing the camera distortion co-efficients
            ``[k0, k1, ..., k13, d]`` where ``k0`` to ``k13`` are distortion co-efficients and d specifies the
            distortion model as defined by the DistortionType enum in camera_info.hpp
    """

    distortion = np.zeros(14 + 1, dtype=np.float32)
    distortion[-1] = 0.0
    distortion = torch.tensor(distortion, dtype=torch.float32)

    return distortion

import xml.etree.ElementTree as ET
def extract_scenario_type(xml_file_path):
    """Extract scenario type from route XML file"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        scenarios = root.findall('.//scenario')
        
        scenario_types = []
        for scenario in scenarios:
            scenario_type = scenario.get('type')
            scenario_name = scenario.get('name')
            if scenario_type:
                scenario_types.append({
                    'name': scenario_name,
                    'type': scenario_type
                })
        
        return scenario_types
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}")
        return []

import hydra
from models.config import TrainConfig
    
@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    # set all seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    cfg.data_module.base_dataset.use_commentary = False
    cfg.data_module.base_dataset.img_shift_augmentation = True
    
    cfg.data_module.base_dataset.use_safety_flag = True

    print('Test Dataset')
    dataset = BaseDataset(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
        if i == 100:
            break

if __name__ == "__main__":
    main()